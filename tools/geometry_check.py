"""
카메라 설치 각도/높이 검토 도구.

기울어진(oblique) 카메라는 두 가지 오차를 만든다. 마운트를 결정하기 전에
이 값들을 보고 각도/높이를 고르는 편이 낫다.

  1) 스케일 변화 (perspective foreshortening)
     가까운 쪽과 먼 쪽의 px/cm가 달라진다.
     단일 px/cm를 쓰는 과거 방식이라면 이 비율이 클수록 거리 계산이 틀어진다.
     현재 런타임은 테이블 평면 Homography를 사용한다.

  2) 시차 오차 (parallax)
     바닥 평면 위로 h만큼 떠 있는 물체(손, 머리)는 바닥에 투영할 때
     실제 위치보다 바깥쪽으로 밀려서 찍힌다.
         변위 = d * h / (H - h)
     여기서 d = 카메라 바로 아래 지점에서의 수평 거리, H = 카메라 높이.
     이게 위험 판정 임계값(15cm)보다 커지면 판정 자체가 무의미해진다.

사용법:
    python tools/geometry_check.py
    python tools/geometry_check.py --tilt 75 --height 250 --area 160x120
"""

from __future__ import annotations

import argparse
import math


def analyze(tilt_deg: float, H: float, area_w: float, area_d: float,
            hfov_deg: float, img_w: int) -> dict:
    """tilt_deg: 수평면 기준 내려다보는 각(90=완전 수직). H: 카메라 높이[cm]."""
    th = math.radians(tilt_deg)
    f_px = img_w / (2 * math.tan(math.radians(hfov_deg) / 2))

    # 광축이 바닥과 만나는 지점 (카메라 바로 아래에서의 거리)
    d_center = H / math.tan(th)

    # 작업영역 근/원 경계
    d_near = d_center - area_d / 2
    d_far = d_center + area_d / 2

    def scales(d):
        r = math.hypot(H, d)
        sin_a = H / r
        s_transverse = f_px / r          # 좌우 방향 px/cm
        s_radial = f_px * sin_a / r      # 깊이 방향 px/cm (단축됨)
        return s_transverse, s_radial

    st_n, sr_n = scales(max(d_near, 1.0))
    st_c, sr_c = scales(d_center)
    st_f, sr_f = scales(d_far)

    def parallax(d, h):
        if h >= H:
            return float("inf")
        return d * h / (H - h)

    return {
        "tilt": tilt_deg, "H": H, "f_px": f_px,
        "d_center": d_center, "d_near": d_near, "d_far": d_far,
        "scale_near": st_n, "scale_center": st_c, "scale_far": st_f,
        "scale_ratio": st_n / st_f if st_f > 0 else float("inf"),
        "aniso_center": st_c / sr_c if sr_c > 0 else float("inf"),
        # 관심 높이별 시차 (작업영역 먼 쪽 = 최악)
        "par_hand_center": parallax(d_center, 20),
        "par_hand_far": parallax(d_far, 20),
        "par_head_center": parallax(d_center, 80),
        "par_head_far": parallax(d_far, 80),
        "par_robot_far": parallax(d_far, 10),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilt", type=float, default=None,
                    help="수평 기준 내려다보는 각도(도). 90=완전 수직")
    ap.add_argument("--height", type=float, default=250.0, help="카메라 높이 [cm]")
    ap.add_argument("--area", type=str, default="160x120", help="작업영역 WxD [cm]")
    ap.add_argument("--hfov", type=float, default=60.0, help="수평 화각 [도]")
    ap.add_argument("--width", type=int, default=1280, help="영상 가로 픽셀")
    args = ap.parse_args()

    aw, ad = (float(x) for x in args.area.lower().split("x"))
    tilts = [args.tilt] if args.tilt else [90, 85, 80, 75, 70, 65, 60]

    print("=" * 78)
    print(f"카메라 설치 검토   높이 {args.height:.0f}cm · 작업영역 {aw:.0f}x{ad:.0f}cm "
          f"· HFOV {args.hfov:.0f}deg · {args.width}px")
    print("=" * 78)
    print()
    print(f"{'각도':>5} {'중심거리':>8} {'px/cm 근':>9} {'px/cm 원':>9} {'스케일차':>8} "
          f"{'왜곡':>6} | {'시차(cm)':>8}")
    print(f"{'':>5} {'':>8} {'':>9} {'':>9} {'근/원':>8} "
          f"{'가로/깊이':>6} | {'손20':>5}{'머리80':>7}")
    print("-" * 78)

    for t in tilts:
        r = analyze(t, args.height, aw, ad, args.hfov, args.width)
        warn = ""
        if r["par_hand_far"] > 15:
            warn = "  <- 손 시차가 DANGER 임계(15cm) 초과"
        elif r["scale_ratio"] > 1.35:
            warn = "  <- 스케일 편차 큼"
        print(f"{t:>4.0f}d {r['d_center']:>7.0f}cm {r['scale_near']:>9.2f} "
              f"{r['scale_far']:>9.2f} {r['scale_ratio']:>7.2f}x "
              f"{r['aniso_center']:>5.2f}x | {r['par_hand_far']:>5.1f} "
              f"{r['par_head_far']:>6.1f}{warn}")

    print()
    print("읽는 법")
    print("  스케일차 근/원 : 1.0에 가까울수록 좋다. 큰 값일수록 테이블 보정이 중요")
    print("  왜곡 가로/깊이 : 깊이 방향이 얼마나 눌려 보이는지. 1.0이면 왜곡 없음")
    print("  시차 손/머리   : 바닥 평면에 투영했을 때 실제 위치에서 밀려나는 거리")
    print("                   손 시차가 15cm(DANGER 임계) 넘으면 판정이 의미를 잃는다")
    print()
    print("대응")
    print("  - 스케일차가 크면: ArUco 4개를 작업영역 네 모서리에 두고 Homography 재보정")
    print("  - 시차가 크면: 각도를 90도에 가깝게 올리거나 카메라를 더 높이 단다")
    print("    시차는 h/(H-h)에 비례하므로 H를 키우는 게 직접적인 해결책")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
