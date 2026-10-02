import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_ai.hand_cup import HandCupEvaluator


def scene(distance, *, grip=False, gaze=False):
    cup = SimpleNamespace(x_cm=0.0, y_cm=0.0)
    human = SimpleNamespace(wrists=[SimpleNamespace(x_cm=distance, y_cm=0.0)])
    hands = SimpleNamespace(any_grasp_ready=grip)
    gaze_info = SimpleNamespace(looking_at_target=gaze)
    return human, cup, hands, gaze_info


def test_close_hand_is_danger_even_when_grip_and_gaze_evidence_present():
    human, cup, hands, gaze = scene(10, grip=True, gaze=True)
    evidence = HandCupEvaluator().update(human, cup, now=1.0, hands=hands, gaze=gaze)
    assert evidence.level == "DANGER"
    assert evidence.grip and evidence.gaze
    assert evidence.distance_cm == 10


def test_approach_and_missing_inputs_reset_history():
    evaluator = HandCupEvaluator()
    human, cup, hands, gaze = scene(70)
    assert evaluator.update(human, cup, now=1.0, hands=hands, gaze=gaze).level == "SAFE"
    human, cup, hands, gaze = scene(30)
    approaching = evaluator.update(human, cup, now=1.1, hands=hands, gaze=gaze)
    assert approaching.level in ("WARN", "DANGER")
    assert approaching.approach_speed_cm_s > 0
    assert evaluator.update(None, cup, now=1.2).level == "UNKNOWN"
    assert evaluator.update(human, cup, now=1.3).approach_speed_cm_s == 0


def test_trigger_wrist_is_selected_by_cup_distance_and_kept():
    left = SimpleNamespace(x_cm=10.0, y_cm=0.0)
    right = SimpleNamespace(x_cm=80.0, y_cm=0.0)
    human = SimpleNamespace(left_wrist=left, right_wrist=right, wrists=[left, right])
    cup = SimpleNamespace(x_cm=0.0, y_cm=0.0)
    evidence = HandCupEvaluator().update(human, cup, now=1.0)
    assert (evidence.trigger_wrist_side, evidence.trigger_wrist_x_cm) == ("LEFT", 10.0)


def test_fast_approach_can_trigger_ttc_danger_from_beyond_warn_distance():
    evaluator = HandCupEvaluator()
    human, cup, _, _ = scene(100)
    evaluator.update(human, cup, now=1.0)
    human, cup, _, _ = scene(50)
    evidence = evaluator.update(human, cup, now=1.1)
    assert evidence.level == "DANGER"
    assert evidence.reason == "TTC"


def test_switching_trigger_wrist_does_not_invent_approach_speed():
    evaluator = HandCupEvaluator()
    left = SimpleNamespace(x_cm=80, y_cm=0)
    right = SimpleNamespace(x_cm=90, y_cm=0)
    human = SimpleNamespace(left_wrist=left, right_wrist=right, wrists=[left, right])
    cup = SimpleNamespace(x_cm=0, y_cm=0)
    evaluator.update(human, cup, now=1.0)
    right.x_cm = 40
    evidence = evaluator.update(human, cup, now=1.1)
    assert evidence.trigger_wrist_side == "RIGHT"
    assert evidence.approach_speed_cm_s == 0
