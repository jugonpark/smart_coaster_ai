"""
MediaPipe Tasks 모델(.task) 다운로드.

Tasks API는 legacy Solutions API와 달리 모델 가중치가 패키지에 포함되어 있지 않다.
Google 공식 저장소에서 한 번 받아 models/ 에 두면 된다.

사용법:
    python tools/download_ai_models.py              # 필요한 것 전부 (pose + hand + face)
    python tools/download_ai_models.py pose         # 특정 모델만
    python tools/download_ai_models.py pose_heavy   # 더 정확한 pose 모델

받는 것:
    pose  : 사람 관절 33점 (손목/팔꿈치/어깨)      약 6MB
    hand  : 손 21점 스켈레톤 + 그립 모양           약 8MB
    face  : 얼굴 468점 + 머리 자세(시선 대용)      약 4MB
"""

import sys
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models"

MODELS = {
    "pose": (
        f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "pose_landmarker_lite.task",
        "사람 관절 33점 (기본)",
    ),
    "pose_full": (
        f"{BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        "pose_landmarker_full.task",
        "사람 관절 - 균형",
    ),
    "pose_heavy": (
        f"{BASE}/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
        "pose_landmarker_heavy.task",
        "사람 관절 - 가장 정확",
    ),
    "hand": (
        f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "hand_landmarker.task",
        "손 21점 스켈레톤 + 그립 모양",
    ),
    "face": (
        f"{BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "face_landmarker.task",
        "얼굴 468점 + 머리 자세 (시선 대용)",
    ),
}

DEFAULT = ["pose", "hand", "face"]


def fetch(key: str, out_dir: Path) -> bool:
    url, fname, desc = MODELS[key]
    out = out_dir / fname

    if out.exists():
        print(f"[{key}] 이미 있음: {fname} ({out.stat().st_size / 1e6:.1f}MB)")
        return True

    print(f"[{key}] {desc}")
    print(f"       {url}")

    def hook(blocks, bs, total):
        if total > 0:
            pct = min(100, blocks * bs * 100 // total)
        if pct % 10 == 0 and pct != getattr(hook, "last_pct", -1):
            print(f"       {pct:3d}%  ({total / 1e6:.1f}MB)", flush=True)
            hook.last_pct = pct

    try:
        urllib.request.urlretrieve(url, out, reporthook=hook)
    except Exception as e:
        out.unlink(missing_ok=True)
        print(f"\n       실패: {e}")
        print(f"       브라우저로 직접 받아 {out} 에 저장해도 됩니다.")
        return False

    print(f"\r       완료: {fname} ({out.stat().st_size / 1e6:.1f}MB)      ")
    return True


def main() -> int:
    keys = [a.lower() for a in sys.argv[1:]] or DEFAULT

    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        print(f"알 수 없는 모델: {', '.join(unknown)}")
        print(f"선택 가능: {', '.join(MODELS)}")
        return 1

    out_dir = Path(__file__).resolve().parent.parent / "models"
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"MediaPipe 모델 다운로드 -> {out_dir}")
    print("=" * 60)

    ok = all([fetch(k, out_dir) for k in keys])

    print()
    print("shared_ai/config.py 대응 항목:")
    print('  POSE_MODEL_PATH = "models/pose_landmarker_lite.task"')
    print('  HAND_MODEL_PATH = "models/hand_landmarker.task"')
    print('  FACE_MODEL_PATH = "models/face_landmarker.task"')
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
