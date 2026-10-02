"""Choose the primary radar or hand-to-cup threat without weakening either."""

from __future__ import annotations

import math

import config
from decision.risk_evaluator import RiskState
from perception.vision_state import sector_for
from planning.transform import world_to_robot


_RANK = {"UNKNOWN": -1, "SAFE": 0, "WARN": 1, "DANGER": 2}


def choose_risk(world, radar_risk: RiskState, *, now: float) -> RiskState:
    evidence = getattr(world, "hand_cup", None)
    if evidence is None or evidence.level not in ("SAFE", "WARN", "DANGER"):
        return radar_risk
    if evidence.timestamp is None or not math.isfinite(evidence.timestamp) or (
        evidence.timestamp > now or now - evidence.timestamp > config.WORLD_STALE_S
    ):
        return RiskState("INVALID", reason="hand wrist stale", timestamp=now)
    hand = RiskState(
        level=evidence.level, distance_cm=evidence.distance_cm,
        approach_speed_cm_s=evidence.approach_speed_cm_s, ttc_s=evidence.ttc_s,
        reason=f"HAND_{evidence.reason} grip={int(evidence.grip)} gaze={int(evidence.gaze)}",
        timestamp=now, source="HAND",
    )
    if hand.level in ("WARN", "DANGER"):
        values = (evidence.trigger_wrist_x_cm, evidence.trigger_wrist_y_cm,
                  world.robot.x_cm, world.robot.y_cm, world.robot.heading_rad)
        if any(value is None or not math.isfinite(value) for value in values):
            return RiskState("INVALID", reason="hand direction invalid", timestamp=now)
        dx, dy = world_to_robot(values[0] - values[2], values[1] - values[3], values[4])
        if math.hypot(dx, dy) < 1e-6:
            return RiskState("INVALID", reason="hand direction unavailable", timestamp=now)
        hand.threat_direction = sector_for(math.atan2(dy, dx))
    if radar_risk.level == "INVALID":
        if config.ALLOW_MOTION_WITHOUT_RADAR and not world.radar_valid and hand.level in ("WARN", "DANGER"):
            return hand
        return radar_risk
    def priority(risk):
        return (_RANK[risk.level],
                -(risk.ttc_s if risk.ttc_s is not None else math.inf),
                -(risk.distance_cm if risk.distance_cm is not None else math.inf))
    return hand if priority(hand) > priority(radar_risk) else radar_risk
