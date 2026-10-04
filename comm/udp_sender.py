"""
전송 계층 - 속도 명령을 JSON으로 직렬화해 UDP로 ESP32에 전송하고, 텔레메트리를 받는다.

왜 UDP인가
  - 제어 명령은 30Hz로 계속 새로 나온다. 하나 잃어도 33ms 뒤 최신 값이 온다.
  - TCP는 재전송/혼잡제어 때문에 오래된 명령이 뒤늦게 도착할 수 있어 오히려 위험하다.
  - 파일이 아니라 소켓으로 흘려보내므로 디스크 I/O 지연도 없다.

프로토콜 (로봇 펌웨어 network_protocol.cpp 의 typed 형식)
    {"type":"cmd_vel","session_id":N,"seq":K,"vx":12.3,"vy":-4.5,"w":0.35,"status":"RUN"}
    {"type":"stop","session_id":N,"seq":K}
    {"type":"heartbeat","session_id":N,"seq":K}
    {"type":"reset_fault","session_id":N,"seq":K}

    - vx/vy [cm/s] 로봇 좌표계 전진/좌측, w [rad/s] 반시계 양수, status RUN|SLOW
    - session_id는 실행할 때마다 새로 뽑는다. 펌웨어는 세션 안에서 seq가 줄어든 패킷을
      stale로 버리므로, 재시작해도 seq가 1부터 다시 시작해 명령이 무시되는 일이 없다.
    - seq는 세션 안에서 모든 종류의 패킷이 공유하는 단조증가 순번.

ESP32 주소 찾기
    펌웨어는 DHCP로 IP를 받는다. config.ESP32_IP가 None이면 브로드캐스트로 STOP을 보내고,
    ESP32가 명령을 받은 상대에게 보내는 텔레메트리(UDP 8889)의 송신 주소를 ESP32 IP로 쓴다.
    주소를 찾기 전에는 움직임 명령을 보내지 않는다.

펌웨어 폴트
    WIFI_LOSS / CMD_TIMEOUT / BAD_PACKET 은 heartbeat로 자동 해제된다(주기적으로 보냄).
    그 외(SPEED_LIMIT, ENCODER_*, PATH_DEVIATION 등)는 래치되며 reset_fault()로만 푼다.

중요: 소켓은 논블로킹으로 둔다.
      네트워크가 막혔을 때 send가 블로킹되면 영상 루프 전체가 멈춘다.
"""

from __future__ import annotations

from typing import Callable

import json
import math
import random
import socket
import time
from dataclasses import dataclass

import config

# heartbeat로 펌웨어가 스스로 해제하는 폴트 (safety_telemetry.cpp isRecoverableFault)
RECOVERABLE_FAULTS = {"WIFI_LOSS", "CMD_TIMEOUT", "BAD_PACKET"}


@dataclass
class CommandPacket:
    type: str          # cmd_vel | stop | heartbeat | reset_fault
    session_id: int
    seq: int
    vx: float = 0.0
    vy: float = 0.0
    w: float = 0.0
    status: str = "STOP"

    def to_json(self) -> str:
        d = {"type": self.type, "session_id": self.session_id, "seq": self.seq}
        if self.type == "cmd_vel":
            # 소수점 3자리로 잘라 패킷 크기를 줄인다 (ESP32 파싱 부담도 감소)
            d.update(vx=round(self.vx, 3), vy=round(self.vy, 3),
                     w=round(self.w, 3), status=self.status)
        return json.dumps(d, separators=(",", ":"))


def clamp_to_firmware(vx: float, vy: float, w: float) -> tuple[float, float, float]:
    """
    펌웨어가 받아들이는 범위로 자른다. 방향은 유지하고 크기만 줄인다.
    펌웨어는 30cm/s·2.0rad/s를 넘는 명령에 SPEED_LIMIT 폴트(래치)를 건다.
    설정 패널에서 최대속도를 올려도 여기서 막힌다.
    """
    lim_v = config.FIRMWARE_LINEAR_LIMIT_CM_S
    lim_w = config.FIRMWARE_ANGULAR_LIMIT_RAD_S
    speed = math.hypot(vx, vy)
    if speed > lim_v:
        k = lim_v / speed
        vx, vy = vx * k, vy * k
    return vx, vy, max(-lim_w, min(lim_w, w))


class UdpSender:
    def __init__(self, ip: str | None = None, port: int | None = None,
                 allow_motion: Callable[[], bool] | None = None) -> None:
        self.allow_motion = allow_motion
        self.port = port or config.ESP32_PORT
        self.esp_ip: str | None = ip or config.ESP32_IP
        self.auto_discover = self.esp_ip is None

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._sock.setblocking(False)

        # 텔레메트리 수신 소켓. 포트가 이미 쓰이면 수신 없이 송신만 한다.
        self._rx: socket.socket | None = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self._rx.bind(("0.0.0.0", config.TELEMETRY_PORT))
            self._rx.setblocking(False)
        except OSError as e:
            print(f"[UDP] 텔레메트리 포트 {config.TELEMETRY_PORT} 사용 불가: {e}")
            self._rx.close()
            self._rx = None

        self.session_id = random.randint(1, 0xFFFFFFFF)
        self._seq = 0
        self._min_interval = 1.0 / config.UDP_SEND_HZ
        self._last_send_t = 0.0
        self._last_heartbeat_t = 0.0
        self._last_discover_t = 0.0

        self.telemetry: dict | None = None
        self.telemetry_t = 0.0
        self._warned_fault: str | None = None
        self._warned_mode = False

        self.sent_count = 0
        self.error_count = 0

        if self.auto_discover:
            print(f"[UDP] ESP32 자동 탐색 (브로드캐스트 {', '.join(config.ESP32_BROADCAST_IPS)}"
                  f" -> 텔레메트리 :{config.TELEMETRY_PORT} 응답 대기)")
            self._discover(time.time())
        else:
            print(f"[UDP] 대상 {self.esp_ip}:{self.port}, {config.UDP_SEND_HZ}Hz")
            self._send_packet(self._packet("stop"))
        print(f"[UDP] session_id={self.session_id}")

    # ------------------------------------------------------------------ 상태
    @property
    def connected(self) -> bool:
        """최근 1초 안에 텔레메트리를 받았는가."""
        return self.telemetry is not None and time.time() - self.telemetry_t < 1.0

    @property
    def fault(self) -> str | None:
        return self.telemetry.get("fault") if self.connected else None

    def link_text(self) -> str:
        """HUD용 한 줄 요약 (영문만)."""
        if self.esp_ip is None:
            return "ESP32 searching..."
        if not self.connected:
            return f"ESP32 {self.esp_ip} no telemetry"
        t = self.telemetry
        return (f"ESP32 {self.esp_ip} {t.get('state')} fault={t.get('fault')} "
                f"rssi={t.get('wifi_rssi')}")

    # ------------------------------------------------------------------ 송신
    def _packet(self, type_: str, **kw) -> CommandPacket:
        self._seq += 1
        return CommandPacket(type=type_, session_id=self.session_id, seq=self._seq, **kw)

    def _send_packet(self, pkt: CommandPacket, addr_ip: str | None = None) -> bool:
        ip = addr_ip or self.esp_ip
        if ip is None:
            return False
        try:
            self._sock.sendto(pkt.to_json().encode("utf-8"), (ip, self.port))
            self.sent_count += 1
            return True
        except (BlockingIOError, OSError) as e:
            # 논블로킹 소켓이 꽉 찼거나 라우팅이 없을 때. 다음 주기에 다시 시도한다.
            self.error_count += 1
            if self.error_count % 60 == 1:
                print(f"[UDP] 송신 실패 ({self.error_count}회): {e}")
            return False

    def _discover(self, now: float) -> None:
        """브로드캐스트 STOP. 정지 명령이라 같은 망의 어떤 로봇이 받아도 안전하다."""
        self._last_discover_t = now
        pkt = self._packet("stop")
        for bcast in config.ESP32_BROADCAST_IPS:
            self._send_packet(pkt, addr_ip=bcast)

    def send(self, vx: float, vy: float, w: float, status: str,
             force: bool = False) -> CommandPacket | None:
        """
        전송 주기를 지키며 명령을 보낸다.
        force=True면 주기를 무시하고 즉시 보낸다 (정지 명령 등 긴급 상황용).
        실제로 보냈으면 패킷을, 건너뛰었으면 None을 반환한다.
        """
        now = time.time()
        if status != "STOP" and self.allow_motion is not None:
            try:
                permitted = bool(self.allow_motion())
            except Exception:
                permitted = False
            if not permitted:
                vx = vy = w = 0.0
                status = "STOP"
                force = True
        self.poll()

        if self.esp_ip is None:
            if now - self._last_discover_t > 0.5:
                self._discover(now)
            return None

        if now - self._last_heartbeat_t > 1.0 / config.HEARTBEAT_HZ:
            self._last_heartbeat_t = now
            self._send_packet(self._packet("heartbeat"))

        if not force and (now - self._last_send_t) < self._min_interval:
            return None

        if status == "STOP" or not all(math.isfinite(v) for v in (vx, vy, w)):
            pkt = self._packet("stop")
        else:
            vx, vy, w = clamp_to_firmware(float(vx), float(vy), float(w))
            pkt = self._packet("cmd_vel", vx=vx, vy=vy, w=w, status=status)

        self._send_packet(pkt)
        self._last_send_t = now
        return pkt

    def send_stop(self) -> None:
        """즉시 정지 명령. 종료 시와 예외 발생 시 여러 번 보낸다 (유실 대비)."""
        for _ in range(5):
            if self.esp_ip is None:
                self._discover(time.time())
            else:
                self._send_packet(self._packet("stop"))
            time.sleep(0.01)

    def reset_fault(self) -> None:
        """
        래치된 폴트 해제 요청. 펌웨어는 reset_fault 뒤 handshake(heartbeat)가 와야 푼다.
        정지 상태에서만 의미가 있으므로 먼저 STOP을 보낸다.
        """
        if self.esp_ip is None:
            print("[UDP] ESP32를 아직 찾지 못해 폴트 리셋을 보낼 수 없습니다.")
            return
        for type_ in ("stop", "reset_fault", "heartbeat"):
            self._send_packet(self._packet(type_))
        self._warned_fault = None
        print("[UDP] 폴트 리셋 요청 전송")

    # ------------------------------------------------------------------ 수신
    def poll(self) -> None:
        """쌓인 텔레메트리를 비우고 최신 값만 남긴다."""
        if self._rx is None:
            return
        for _ in range(32):
            try:
                data, addr = self._rx.recvfrom(2048)
            except (BlockingIOError, OSError):
                break
            try:
                msg = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(msg, dict) or msg.get("type") != "telemetry":
                continue
            if self.auto_discover and addr[0] != self.esp_ip:
                print(f"[UDP] ESP32 발견: {addr[0]}:{self.port}")
                self.esp_ip = addr[0]
            elif addr[0] != self.esp_ip:
                continue
            self.telemetry = msg
            self.telemetry_t = time.time()
        self._report_state()

    def _report_state(self) -> None:
        if not self.connected:
            return
        fault = self.telemetry.get("fault", "NONE")
        if fault != "NONE" and fault != self._warned_fault:
            self._warned_fault = fault
            if fault in RECOVERABLE_FAULTS:
                print(f"[ESP32] 폴트 {fault} - heartbeat로 자동 해제 시도")
                self._last_heartbeat_t = 0.0
            else:
                print(f"[ESP32] 폴트 {fault} (래치) - 원인 확인 후 'r' 키로 리셋")
        elif fault == "NONE" and self._warned_fault is not None:
            print("[ESP32] 폴트 해제")
            self._warned_fault = None

        if self.telemetry.get("mode") == "MANUAL_TEST" and not self._warned_mode:
            self._warned_mode = True
            print("[ESP32] MANUAL_TEST 모드입니다. 시리얼 모니터에서 AUTO 를 입력하세요.")

    def close(self) -> None:
        self.send_stop()
        self._sock.close()
        if self._rx is not None:
            self._rx.close()
        print(f"[UDP] 종료. 전송 {self.sent_count}건, 실패 {self.error_count}건")
