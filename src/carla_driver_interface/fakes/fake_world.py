# SPDX-License-Identifier: Apache-2.0
"""A CARLA-free :class:`WorldAdapter` for tests and demos.

The `carla` module cannot be installed in CI, so the closed loop would otherwise
be untestable.  :class:`FakeWorld` implements the same protocol on top of a
kinematic bicycle model, which is enough to exercise every step of the
runtime--driver exchange: session setup, image submission, egomotion, route,
the ``drive`` call, control application and metrics.

It is deliberately *not* a physics model.  Anything relying on real vehicle
dynamics belongs in a test against a real CARLA server.
"""

from __future__ import annotations

import math

import numpy as np

from carla_driver_interface.geometry import Pose
from carla_driver_interface.grpc_api import (
    RendererData,
    TrafficLight,
    TrafficLightState,
    Weather,
)
from carla_driver_interface.runtime.config import LidarConfig, RuntimeConfig, ScenarioSpec
from carla_driver_interface.runtime.control import VehicleCommand, steer_angle
from carla_driver_interface.runtime.conversions import (
    available_camera,
    camera_pose_in_rig,
    seconds_to_us,
    sensor_pose_in_rig,
)
from carla_driver_interface.runtime.images import encode_rgb
from carla_driver_interface.runtime.world import (
    CameraCapture,
    EgoState,
    LidarCapture,
    RolloutEvents,
    WorldSetup,
    WorldSnapshot,
)

__all__ = ["FakeWorld", "straight_then_curve_route"]

#: Azimuth samples per channel in a synthetic sweep. Bounded so CI stays fast
#: whatever ``points_per_second`` a test configures.
_FAKE_LIDAR_MAX_AZIMUTHS = 360


def straight_then_curve_route(
    straight_m: float = 40.0,
    radius_m: float = 30.0,
    turn_rad: float = math.pi / 2,
    resolution_m: float = 1.0,
) -> np.ndarray:
    """A route that goes straight, then turns left at constant radius.

    Exercises both the lateral controller and the policy's curvature slowdown.
    """
    points = [
        np.array([distance, 0.0, 0.0]) for distance in np.arange(0.0, straight_m, resolution_m)
    ]
    centre = np.array([straight_m, radius_m, 0.0])
    n_arc = max(2, int(round(radius_m * turn_rad / resolution_m)))
    for i in range(n_arc + 1):
        theta = turn_rad * i / n_arc
        points.append(centre + radius_m * np.array([math.sin(theta), -math.cos(theta), 0.0]))
    return np.stack(points)


class FakeWorld:
    """Kinematic bicycle model standing in for a CARLA server."""

    def __init__(
        self,
        config: RuntimeConfig,
        scenario: ScenarioSpec | None = None,
        route_in_local: np.ndarray | None = None,
        max_speed_mps: float = 20.0,
        max_accel_mps2: float = 4.0,
        max_brake_mps2: float = 8.0,
        off_route_tolerance_m: float = 6.0,
        opendrive: str | None = None,
        traffic_lights: list[TrafficLight] | None = None,
    ) -> None:
        self.config = config
        self.scenario = scenario or ScenarioSpec(map_name="FakeTown", name="straight_then_curve")
        self._route = (
            straight_then_curve_route()
            if route_in_local is None
            else np.asarray(route_in_local, dtype=np.float64).reshape(-1, 3)
        )
        self.max_speed_mps = max_speed_mps
        self.max_accel_mps2 = max_accel_mps2
        self.max_brake_mps2 = max_brake_mps2
        self.off_route_tolerance_m = off_route_tolerance_m
        #: Handed to the runtime as the world's map, for ``map_dir``.
        self.opendrive = opendrive
        #: Reported every step when the runtime writes a map; tests set states.
        self.traffic_lights = list(traffic_lights or [])

        self._x = float(self._route[0][0])
        self._y = float(self._route[0][1])
        self._yaw = math.atan2(
            float(self._route[1][1] - self._route[0][1]),
            float(self._route[1][0] - self._route[0][0]),
        )
        self._speed = 0.0
        self._yaw_rate = 0.0
        self._acceleration = 0.0

        self._frame_id = 0
        self._time_s = 0.0
        self._command = VehicleCommand()
        self._events = RolloutEvents()
        #: Every ego state produced so far; tests assert on the driven path.
        self.history: list[EgoState] = []
        self._lidar_poses = {
            lidar.logical_id: sensor_pose_in_rig(
                lidar.x, lidar.y, lidar.z, lidar.pitch_deg, lidar.yaw_deg, lidar.roll_deg, 0.0
            )
            for lidar in config.lidars
        }
        # Flat ground and a fixed mount: the sweep never changes, so it is cast once.
        self._sweeps = {lidar.logical_id: self._cast_sweep(lidar) for lidar in config.lidars}

    # -- WorldAdapter ------------------------------------------------------

    def setup(self) -> WorldSetup:
        cameras = [
            available_camera(
                logical_id=cam.logical_id,
                width=cam.width,
                height=cam.height,
                horizontal_fov_deg=cam.fov_deg,
                # Through the same conversion the real adapter uses, so a
                # mount rotation cannot be honoured in CARLA and dropped in CI.
                # The fake rig has no actor/rig offset to reconcile.
                pose_in_rig=camera_pose_in_rig(
                    cam.x, cam.y, cam.z, cam.pitch_deg, cam.yaw_deg, cam.roll_deg, 0.0
                ),
            )
            for cam in self.config.cameras
        ]
        return WorldSetup(
            map_name=self.scenario.map_name,
            cameras=cameras,
            rear_axle_offset_m=0.0,
            route_in_local=self._route,
            opendrive=self.opendrive,
        )

    def apply_control(self, command: VehicleCommand) -> None:
        self._command = command

    def tick(self, capture: bool = True) -> WorldSnapshot:
        dt = self.config.fixed_delta_s
        self._integrate(dt)

        self._frame_id += 1
        self._time_s += dt
        timestamp_us = seconds_to_us(self._time_s, self.config.epoch_offset_us)

        ego = self._ego_state(timestamp_us)
        self.history.append(ego)
        self._update_events(ego)

        return WorldSnapshot(
            frame_id=self._frame_id,
            timestamp_us=timestamp_us,
            ego=ego,
            captures=self._captures(timestamp_us) if capture else [],
            lidar=[]
            if not capture
            else [
                LidarCapture(
                    logical_id=lidar.logical_id,
                    timestamp_us=timestamp_us,
                    pose_in_rig=self._lidar_poses[lidar.logical_id],
                    points_in_sensor=self._sweeps[lidar.logical_id],
                )
                for lidar in self.config.lidars
            ],
        )

    def environment(self, snapshot: WorldSnapshot) -> RendererData:
        return RendererData(
            snapshot_timestamp_us=snapshot.timestamp_us,
            frame_id=snapshot.frame_id,
            map_name=self.scenario.map_name,
            weather=Weather(sun_altitude_angle=45.0),
            ego_traffic_light=TrafficLightState.TRAFFIC_LIGHT_STATE_NONE,
            ego_traffic_light_distance_m=-1.0,
            speed_limit_mps=self.max_speed_mps,
            traffic_lights=self.traffic_lights if self.config.map_dir else [],
        )

    def events(self) -> RolloutEvents:
        return self._events

    def close(self) -> None:
        return None

    # -- model -------------------------------------------------------------

    def _integrate(self, dt: float) -> None:
        """Advance the bicycle model by ``dt`` under the latched command."""
        command = self._command
        accel = command.throttle * self.max_accel_mps2 - command.brake * self.max_brake_mps2
        # Rolling resistance, so coasting decays rather than holding forever.
        accel -= 0.05 * self._speed

        previous_speed = self._speed
        self._speed = float(np.clip(self._speed + accel * dt, 0.0, self.max_speed_mps))
        self._acceleration = (self._speed - previous_speed) / dt

        # CARLA steers positive to the right; the local frame is positive to the
        # left, hence the sign flip -- the same convention the controller uses.
        wheel_angle = -steer_angle(command.steer, self.config.control)
        self._yaw_rate = self._speed * math.tan(wheel_angle) / self.config.control.wheelbase_m

        self._yaw += self._yaw_rate * dt
        self._x += self._speed * math.cos(self._yaw) * dt
        self._y += self._speed * math.sin(self._yaw) * dt

    def _ego_state(self, timestamp_us: int) -> EgoState:
        return EgoState(
            timestamp_us=timestamp_us,
            pose_local_to_rig=Pose.from_xyz_yaw(self._x, self._y, 0.0, self._yaw),
            # A bicycle model has no lateral slip, so rig-frame velocity is
            # purely longitudinal.
            linear_velocity_in_rig=np.array([self._speed, 0.0, 0.0]),
            angular_velocity_in_rig=np.array([0.0, 0.0, self._yaw_rate]),
            linear_acceleration_in_rig=np.array([self._acceleration, 0.0, 0.0]),
        )

    def _update_events(self, ego: EgoState) -> None:
        """Treat leaving the route corridor as a lane invasion."""
        position = ego.pose_local_to_rig.position[:2]
        distance = float(np.min(np.linalg.norm(self._route[:, :2] - position, axis=1)))
        if distance > self.off_route_tolerance_m:
            self._events.lane_invasions += 1

    def _captures(self, timestamp_us: int) -> list[CameraCapture]:
        """Synthesise a frame per camera: a horizon gradient, so it is not blank."""
        captures = []
        for cam in self.config.cameras:
            rgb = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
            horizon = cam.height // 2
            rgb[:horizon] = (110, 160, 220)  # sky
            rgb[horizon:] = (70, 70, 70)  # road
            # A moving marker keeps successive frames distinguishable.
            column = int((self._x * 4) % cam.width)
            rgb[:, column] = (255, 0, 0)
            captures.append(
                CameraCapture(
                    logical_id=cam.logical_id,
                    frame_start_us=timestamp_us,
                    frame_end_us=timestamp_us,
                    image_bytes=encode_rgb(
                        rgb, self.config.image_format, self.config.image_quality
                    ),
                )
            )
        return captures

    # -- LiDAR ---------------------------------------------------------------

    def _cast_sweep(self, lidar: LidarConfig) -> np.ndarray:
        """A flat-ground sweep: every downward beam that lands within range.

        Cast in the rig frame, where the ground is z = 0, then expressed in
        CARLA's left-handed sensor frame -- the buffer a real sensor delivers --
        so it reaches the rig through the same
        :func:`~carla_driver_interface.runtime.conversions.lidar_points_to_rig`
        a real sweep takes. A level, centred mount sweeps a y-symmetric ring
        that a lost mirror would leave unchanged; a rolled or offset mount does
        not (``tests/test_lidar.py`` uses one).
        """
        pose = self._lidar_poses[lidar.logical_id]
        per_sweep = max(1, int(lidar.points_per_second * self.config.fixed_delta_s))
        azimuths = int(np.clip(per_sweep // lidar.channels, 1, _FAKE_LIDAR_MAX_AZIMUTHS))
        elevation = np.radians(
            np.linspace(lidar.lower_fov_deg, lidar.upper_fov_deg, lidar.channels)
        )
        azimuth = np.linspace(0.0, 2.0 * np.pi, azimuths, endpoint=False)
        el, az = np.meshgrid(elevation, azimuth, indexing="ij")
        direction_sensor = np.stack(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1
        ).reshape(-1, 3)
        direction_rig = direction_sensor @ pose.rotation_matrix.T
        height = float(pose.position[2])
        down = direction_rig[:, 2] < -1e-6
        distance = np.full(len(direction_rig), np.inf)
        distance[down] = height / -direction_rig[down, 2]
        hit = distance <= lidar.range_m
        points_rig = pose.position + direction_rig[hit] * distance[hit, None]

        points_sensor = pose.inverse().transform_points(points_rig)
        points_sensor[:, 1] = -points_sensor[:, 1]  # into CARLA's left-handed frame
        intensity = np.full((len(points_sensor), 1), 0.5)
        return np.concatenate([points_sensor, intensity], axis=1).astype(np.float32)
