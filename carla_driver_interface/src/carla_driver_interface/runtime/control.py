# SPDX-License-Identifier: Apache-2.0
"""Trajectory tracking: alpasim's ``VDCService`` replaced by a local controller.

alpasim hands the driver's plan to a separate vehicle-dynamics-and-controller
service over gRPC (``controller.proto``).  CARLA already simulates the vehicle,
so all that is missing is the controller itself: turn a planned trajectory into
throttle / brake / steer.

Lateral control is pure pursuit, trimmed by feedback on the measured yaw rate;
longitudinal control is a PI(D) on speed. Neither touches the ``carla``
module, so both are unit-testable without a simulator --
:class:`VehicleCommand` is a plain dataclass the world adapter translates into
``carla.VehicleControl``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from carla_driver_interface import polyline
from carla_driver_interface.geometry import Pose, Trajectory

__all__ = [
    "ControlConfig",
    "TrajectoryFollower",
    "VehicleCommand",
    "steer_angle",
    "steer_command",
]


@dataclass(frozen=True)
class ControlConfig:
    """Tuning for :class:`TrajectoryFollower`."""

    #: Lookahead grows with speed: ``clamp(gain * speed, min, max)``.
    lookahead_gain_s: float = 0.6
    min_lookahead_m: float = 4.0
    max_lookahead_m: float = 20.0
    #: Distance from the rig origin (rear axle) to the front axle.
    wheelbase_m: float = 2.8
    #: How CARLA's [-1, 1] steer turns the wheels:
    #: ``angle = max_steer_angle * |steer| ** steer_exponent``. CARLA 0.10's
    #: Chaos vehicles answer quadratically, whatever ``max_steer_angle`` the
    #: wheel reports (70 degrees): measured on the MKZ at 2 and 8 m/s, steer
    #: 0.1 turns like 0.56 degrees, 0.3 like 5.0, 0.5 like 13.9 -- 56 * steer**2
    #: within a few percent up to 0.5, at every speed. A linear map asks for a
    #: tenth of the needed angle on a gentle curve. 1.0 is a linear map.
    max_steer_angle_rad: float = math.radians(56.0)
    steer_exponent: float = 2.0
    #: Rate limit on the normalised steering command, per second.
    max_steer_rate: float = 4.0
    #: Integral feedback on the yaw rate. Pure pursuit asks for the yaw rate
    #: ``speed * curvature``; this trims the steering angle until the vehicle
    #: delivers it, absorbing what the steering map and the bicycle model
    #: miss (understeer, a different vehicle's steering response). The error is
    #: taken as the steering angle the bicycle model would need for the
    #: missing yaw rate, so the gain is unitless and the same at every speed.
    #: Integral only: the vehicle answers within a tick, and a proportional
    #: term on top of that delay oscillates. Zero steers open-loop.
    yaw_rate_ki: float = 3.0
    #: Clamp on the integrated trim, in radians of steering.
    yaw_rate_trim_limit_rad: float = math.radians(25.0)
    #: Below this speed the yaw rate says little about the steering, so the
    #: trim is held rather than integrated.
    yaw_rate_min_speed_mps: float = 1.0

    speed_kp: float = 0.6
    speed_ki: float = 0.15
    speed_kd: float = 0.05
    #: Integrator clamp, in normalised command units.
    integral_limit: float = 1.0

    #: Below this target speed the controller brakes to a stop instead of
    #: modulating throttle, so it settles cleanly at red lights.
    stop_speed_mps: float = 0.2
    stop_brake: float = 0.6


def steer_command(angle_rad: float, config: ControlConfig) -> float:
    """The [-1, 1] steer that turns the wheels by ``angle_rad``, same sign."""
    ratio = min(1.0, abs(angle_rad) / config.max_steer_angle_rad)
    return math.copysign(ratio ** (1.0 / config.steer_exponent), angle_rad)


def steer_angle(command: float, config: ControlConfig) -> float:
    """Inverse of :func:`steer_command`: the wheel angle a steer gives."""
    ratio = min(1.0, abs(command)) ** config.steer_exponent
    return math.copysign(ratio * config.max_steer_angle_rad, command)


@dataclass(frozen=True)
class VehicleCommand:
    """Actuation request, in CARLA's normalised units."""

    throttle: float = 0.0
    steer: float = 0.0
    brake: float = 0.0
    hand_brake: bool = False
    reverse: bool = False

    #: Diagnostics, surfaced in rollout metrics rather than sent to CARLA.
    target_speed_mps: float = 0.0
    #: Lateral offset of the pure-pursuit target, in the rig frame. Positive is
    #: to the left. This is what the steering command reacts to. It is *not* a
    #: tracking error: the driver anchors its plan on the ego, so the ego's
    #: offset from its own plan is structurally zero -- the runtime measures
    #: tracking against the route instead (``route_lateral_error_m``).
    lookahead_lateral_offset_m: float = 0.0


class TrajectoryFollower:
    """Stateful tracker: feed it plans, get actuation.

    One instance per rollout -- it carries the speed integrator and the previous
    steering command.
    """

    def __init__(self, config: ControlConfig | None = None) -> None:
        self.config = config or ControlConfig()
        self._integral = 0.0
        self._previous_speed_error = 0.0
        self._previous_steer = 0.0
        self._yaw_rate_trim = 0.0
        self._asked_curvature: float | None = None

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_speed_error = 0.0
        self._previous_steer = 0.0
        self._yaw_rate_trim = 0.0
        self._asked_curvature = None

    def step(
        self,
        plan_in_local: Trajectory,
        pose_local_to_rig: Pose,
        current_speed_mps: float,
        dt_s: float,
        yaw_rate_rps: float | None = None,
    ) -> VehicleCommand:
        """Compute actuation for one control step.

        Args:
            plan_in_local: The driver's plan, in the ``local`` frame.
            pose_local_to_rig: Where the ego actually is now.
            current_speed_mps: Longitudinal speed.
            dt_s: Time since the previous call.
            yaw_rate_rps: The ego's measured yaw rate, positive to the left.
                ``None`` steers on pure pursuit alone.
        """
        if len(plan_in_local) < 2 or dt_s <= 0.0:
            # Nothing to track: coast, holding the last steering angle.
            return VehicleCommand(throttle=0.0, steer=self._previous_steer, brake=0.3)

        # Only positions are needed downstream, so rotate the points rather
        # than composing 40-odd quaternions whose rotations are then discarded.
        points = pose_local_to_rig.inverse().transform_points(plan_in_local.positions)
        arc = polyline.arc_lengths(points)

        target_speed = _plan_speed(plan_in_local.timestamps_us, arc)
        steer, lateral_offset = self._lateral(points, arc, current_speed_mps, dt_s, yaw_rate_rps)
        throttle, brake = self._longitudinal(target_speed, current_speed_mps, dt_s)

        return VehicleCommand(
            throttle=throttle,
            steer=steer,
            brake=brake,
            target_speed_mps=target_speed,
            lookahead_lateral_offset_m=lateral_offset,
        )

    # -- lateral -----------------------------------------------------------

    def _lateral(
        self,
        points_in_rig: np.ndarray,
        arc: np.ndarray,
        speed_mps: float,
        dt_s: float,
        yaw_rate_rps: float | None,
    ) -> tuple[float, float]:
        cfg = self.config
        lookahead = min(
            cfg.max_lookahead_m,
            max(cfg.min_lookahead_m, cfg.lookahead_gain_s * speed_mps),
        )
        target = polyline.sample(points_in_rig, arc, lookahead)

        # Pure pursuit in the rig frame: the rig origin is the rear axle, so
        # the target's bearing `alpha` is measured straight off its coordinates.
        distance = float(np.linalg.norm(target[:2]))
        if distance < 1e-3:
            curvature = 0.0
        else:
            alpha = math.atan2(float(target[1]), float(target[0]))
            curvature = 2.0 * math.sin(alpha) / distance
        steer_angle = math.atan(cfg.wheelbase_m * curvature)
        if yaw_rate_rps is not None:
            steer_angle += self._yaw_rate_feedback(speed_mps, yaw_rate_rps, curvature, dt_s)

        # CARLA steers positive to the right; the rig frame is positive to the
        # left, so the command is the negated steering angle.
        command = -steer_command(steer_angle, cfg)

        max_delta = cfg.max_steer_rate * dt_s
        command = float(
            np.clip(command, self._previous_steer - max_delta, self._previous_steer + max_delta)
        )
        self._previous_steer = command
        return command, float(target[1])

    def _yaw_rate_feedback(
        self, speed_mps: float, yaw_rate_rps: float, curvature: float, dt_s: float
    ) -> float:
        """Steering trim, in radians, that closes the gap to the asked yaw rate."""
        cfg = self.config
        # The measured yaw rate answers the previous step's command, so it is
        # held against the curvature asked then; against this step's, every
        # change in the plan would read as a steering error for one step.
        asked, self._asked_curvature = self._asked_curvature, curvature
        if asked is None or speed_mps < cfg.yaw_rate_min_speed_mps:
            return self._yaw_rate_trim
        # The yaw-rate error as a steering angle: what the bicycle model says
        # it would take to turn by the missing yaw rate.
        error = cfg.wheelbase_m * (asked - yaw_rate_rps / speed_mps)
        limit = cfg.yaw_rate_trim_limit_rad
        self._yaw_rate_trim = float(
            np.clip(self._yaw_rate_trim + cfg.yaw_rate_ki * error * dt_s, -limit, limit)
        )
        return self._yaw_rate_trim

    # -- longitudinal ------------------------------------------------------

    def _longitudinal(
        self, target_speed: float, current_speed: float, dt_s: float
    ) -> tuple[float, float]:
        cfg = self.config
        if target_speed <= cfg.stop_speed_mps:
            self._integral = 0.0
            self._previous_speed_error = 0.0
            return 0.0, cfg.stop_brake

        error = target_speed - current_speed
        derivative = (error - self._previous_speed_error) / dt_s
        self._previous_speed_error = error

        candidate = self._integral + error * dt_s
        raw = cfg.speed_kp * error + cfg.speed_ki * candidate + cfg.speed_kd * derivative
        # Anti-windup: only integrate while the command is not saturated, or
        # while the error pushes it back out of saturation.
        if -1.0 < raw < 1.0 or (raw >= 1.0 and error < 0.0) or (raw <= -1.0 and error > 0.0):
            self._integral = float(np.clip(candidate, -cfg.integral_limit, cfg.integral_limit))
        command = cfg.speed_kp * error + cfg.speed_ki * self._integral + cfg.speed_kd * derivative

        if command >= 0.0:
            return float(min(command, 1.0)), 0.0
        return 0.0, float(min(-command, 1.0))


# ---------------------------------------------------------------------------
# Plan helpers
# ---------------------------------------------------------------------------


def _plan_speed(timestamps_us: list[int], arc: np.ndarray) -> float:
    """Speed the plan implies over its first second (or its whole length).

    Averaging over a window rather than the first pair keeps the target steady
    when the driver emits a plan at a finer resolution than the control step.
    """
    t0 = timestamps_us[0]
    end = len(timestamps_us) - 1
    for i, ts in enumerate(timestamps_us):
        if ts - t0 >= 1_000_000:
            end = i
            break
    dt_s = (timestamps_us[end] - t0) / 1e6
    if dt_s <= 0.0:
        return 0.0
    return float(arc[end]) / dt_s
