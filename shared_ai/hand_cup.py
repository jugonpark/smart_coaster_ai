"""Hand-to-cup proximity evidence; no robot command or safety override.

Coordinates must already be calibrated to the same table plane in centimeters.
Grip and gaze are reported as evidence and never lower the risk level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class HandCupEvidence:
    level: str = "UNKNOWN"
    distance_cm: float | None = None
    approach_speed_cm_s: float = 0.0
    ttc_s: float | None = None
    trigger_wrist_x_cm: float | None = None
    trigger_wrist_y_cm: float | None = None
    trigger_wrist_side: str | None = None
    cup_x_cm: float | None = None
    cup_y_cm: float | None = None
    grip: bool = False
    gaze: bool = False
    timestamp: float | None = None
    reason: str = "hand or cup unavailable"

    @property
    def wrist_x_cm(self) -> float | None:
        return self.trigger_wrist_x_cm

    @property
    def wrist_y_cm(self) -> float | None:
        return self.trigger_wrist_y_cm


class HandCupEvaluator:
    def __init__(self, *, danger_cm: float = 15.0, warn_cm: float = 35.0,
                 approach_cm_s: float = 25.0, danger_ttc_s: float = 0.8,
                 warn_ttc_s: float = 1.6) -> None:
        self.danger_cm = danger_cm
        self.warn_cm = warn_cm
        self.approach_cm_s = approach_cm_s
        self.danger_ttc_s = danger_ttc_s
        self.warn_ttc_s = warn_ttc_s
        self._last_distance: float | None = None
        self._last_at: float | None = None
        self._last_side: str | None = None

    def update(self, human, cup, *, now: float, hands=None, gaze=None) -> HandCupEvidence:
        wrists = getattr(human, "wrists", ()) if human is not None else ()
        if cup is None or not wrists or not math.isfinite(now):
            self._last_distance = self._last_at = self._last_side = None
            return HandCupEvidence(timestamp=now)
        try:
            cup_x, cup_y = float(cup.x_cm), float(cup.y_cm)
            candidates = [(math.hypot(float(w.x_cm) - cup_x, float(w.y_cm) - cup_y), w)
                          for w in wrists]
            distance, wrist = min(candidates, key=lambda item: item[0])
            if not all(math.isfinite(value) for value in
                       (cup_x, cup_y, distance, wrist.x_cm, wrist.y_cm)):
                raise ValueError("non-finite hand/cup coordinate")
        except (AttributeError, TypeError, ValueError, OverflowError):
            self._last_distance = self._last_at = self._last_side = None
            return HandCupEvidence(timestamp=now, reason="invalid hand or cup coordinate")
        side = ("LEFT" if wrist is getattr(human, "left_wrist", None) else
                "RIGHT" if wrist is getattr(human, "right_wrist", None) else "UNKNOWN")
        speed = 0.0
        if self._last_at is not None and side == self._last_side and now > self._last_at:
            speed = max(0.0, (self._last_distance - distance) / (now - self._last_at))
        self._last_distance, self._last_at, self._last_side = distance, now, side
        ttc = distance / speed if speed > 1.0 else None
        if distance < self.danger_cm or (speed > self.approach_cm_s and
                                         ttc is not None and ttc < self.danger_ttc_s):
            level = "DANGER"
        elif distance < self.warn_cm or (speed > self.approach_cm_s and
                                         ttc is not None and ttc < self.warn_ttc_s):
            level = "WARN"
        else:
            level = "SAFE"
        reason = ("DISTANCE" if distance < (self.danger_cm if level == "DANGER" else self.warn_cm)
                  else "TTC" if level != "SAFE" else "normal")
        return HandCupEvidence(
            level=level, distance_cm=distance, approach_speed_cm_s=speed, ttc_s=ttc,
            trigger_wrist_x_cm=float(wrist.x_cm), trigger_wrist_y_cm=float(wrist.y_cm),
            trigger_wrist_side=side, cup_x_cm=cup_x, cup_y_cm=cup_y,
            grip=bool(getattr(hands, "any_grasp_ready", False)),
            gaze=bool(getattr(gaze, "looking_at_target", False)), timestamp=now,
            reason=reason,
        )
