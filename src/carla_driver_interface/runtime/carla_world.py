# SPDX-License-Identifier: Apache-2.0
"""The :class:`~carla_driver_interface.runtime.world.WorldAdapter` backed by CARLA.

Split out from the contract module so that a process without CARLA -- CI running
against the fake, or a driver-only install -- never imports this file, and so
that the seam stays readable next to its one large implementation.

Written against the API surface shared by CARLA 0.9.x and 0.10.x; the few places
where they diverge are marked and probed defensively rather than branched on a
version string, because forks report versions inconsistently.
"""

from __future__ import annotations

import logging
import math
import queue
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from carla_driver_interface.geometry import Pose
from carla_driver_interface.grpc_api import (
    AvailableCamera,
    RendererData,
)
from carla_driver_interface.runtime.config import (
    CameraConfig,
    LidarConfig,
    RuntimeConfig,
    ScenarioSpec,
)
from carla_driver_interface.runtime.control import VehicleCommand
from carla_driver_interface.runtime.conversions import (
    available_camera,
    camera_pose_in_rig,
    carla_angular_velocity_to_local,
    carla_transform_to_pose,
    carla_vector_to_local,
    rig_pose_from_actor_transform,
    seconds_to_us,
    sensor_pose_in_rig,
    vector_local_to_rig,
    waypoint_to_local,
)
from carla_driver_interface.runtime.ground_truth import CarlaGroundTruth
from carla_driver_interface.runtime.images import encode_bgra
from carla_driver_interface.runtime.lanes import carla_lane_key
from carla_driver_interface.runtime.world import (
    CameraCapture,
    EgoState,
    LidarCapture,
    RolloutEvents,
    WorldSetup,
    WorldSnapshot,
)

logger = logging.getLogger(__name__)

__all__ = ["CarlaWorldAdapter", "load_carla_module"]


def load_carla_module(python_path: str | None = None) -> Any:
    """Import ``carla``, optionally from an out-of-tree PythonAPI.

    CARLA 0.10.x is not published on PyPI; it ships its own PythonAPI directory.
    ``python_path`` prepends that directory to ``sys.path`` before importing.
    """
    if python_path:
        import sys

        if python_path not in sys.path:
            sys.path.insert(0, python_path)
    try:
        import carla
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the `carla` module is not importable. Install the extra "
            "(`uv sync --extra carla`, CARLA 0.9.x) or point --carla-python-path "
            "at a 0.10.x PythonAPI directory."
        ) from exc
    return carla


@dataclass(frozen=True)
class _RawFrame:
    """An un-encoded CARLA capture, waiting to be picked up by the runtime."""

    frame: int
    timestamp_us: int
    width: int
    height: int
    bgra: bytes


@dataclass(frozen=True)
class _RawSweep:
    """An un-converted CARLA LiDAR measurement, in the sensor's own frame."""

    frame: int
    timestamp_us: int
    points_xyzi: np.ndarray


#: How long a tick waits for a sensor's measurement of its own frame. In
#: synchronous mode measurements are delivered on the client's sensor thread
#: *after* `tick()` returns, so reading a queue straight away misses them now
#: and then -- measured on Town10HD_Opt, often enough for a LiDAR-reading
#: driver to fail a 250-step rollout -- and a late camera frame is a stale one.
_SENSOR_WAIT_S = 2.0


def _take_frame(pending: queue.Queue, frame_id: int, sensor: str, deadline: float) -> Any:
    """This tick's measurement, waiting for it until ``deadline`` (``time.monotonic``).

    Any backlog is discarded: it is older than this tick. When the measurement
    does not arrive in time, ``None`` -- an older frame would be stale, which is
    exactly what must not be submitted. One deadline is shared by every sensor
    of a tick, so a tick waits at most ``_SENSOR_WAIT_S`` however many miss.
    """
    newest = None
    dropped = -1
    while True:
        try:
            newest = pending.get_nowait()
            dropped += 1
        except queue.Empty:
            break
    try:
        while newest is None or newest.frame < frame_id:
            newest = pending.get(timeout=max(0.0, deadline - time.monotonic()))
    except queue.Empty:
        logger.warning("no measurement from %s for frame %d in time; skipped", sensor, frame_id)
        return None
    if dropped > 0:
        logger.debug("dropped %d stale measurements from %s", dropped, sensor)
    if newest.frame != frame_id:
        # Only when another client ticks the world: the measurement is newer
        # than the snapshot it is submitted with.
        logger.debug("%s delivered frame %d for tick %d", sensor, newest.frame, frame_id)
    return newest


class CarlaWorldAdapter:
    """Drives a real CARLA server.

    Written against the API surface shared by 0.9.x and 0.10.x; the few places
    where they diverge are marked and probed defensively rather than branched on
    a version string, because forks report versions inconsistently.
    """

    def __init__(
        self,
        config: RuntimeConfig,
        scenario: ScenarioSpec,
        carla_python_path: str | None = None,
    ) -> None:
        self.config = config
        self.scenario = scenario
        self._carla = load_carla_module(carla_python_path)
        self._rng = random.Random(config.seed)

        self._client: Any = None
        self._world: Any = None
        self._map: Any = None
        self._traffic_manager: Any = None
        self._original_settings: Any = None

        self._ego: Any = None
        self._sensors: list[Any] = []
        self._background: list[Any] = []
        self._frame_queues: dict[str, queue.Queue] = {}
        self._sweep_queues: dict[str, queue.Queue] = {}
        self._lidar_poses: dict[str, Pose] = {}
        self._route_lane_ids: list[str] = []
        self._events = RolloutEvents()
        self._rear_axle_offset_m = 0.0
        self._pending_control: VehicleCommand | None = None
        self._ground_truth_reader: CarlaGroundTruth | None = None

    # -- setup -------------------------------------------------------------

    def setup(self) -> WorldSetup:
        carla = self._carla
        self._client = carla.Client(self.config.carla_host, self.config.carla_port)
        self._client.set_timeout(self.config.carla_timeout_s)

        self._world = self._client.load_world(self.scenario.map_name)
        self._map = self._world.get_map()

        self._original_settings = self._world.get_settings()
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config.fixed_delta_s
        self._world.apply_settings(settings)

        self._traffic_manager = self._client.get_trafficmanager(self.config.traffic_manager_port)
        self._traffic_manager.set_synchronous_mode(True)
        self._traffic_manager.set_random_device_seed(self.config.seed)

        if self.scenario.weather_preset:
            self._apply_weather(self.scenario.weather_preset)

        spawn_transform = self._spawn_ego()
        self._rear_axle_offset_m = self._resolve_rear_axle_offset()
        cameras = self._spawn_cameras()
        self._spawn_lidars()
        self._spawn_collision_sensors()
        self._spawn_background_traffic()

        route = self._build_route(spawn_transform)

        return WorldSetup(
            map_name=self.scenario.map_name,
            cameras=cameras,
            rear_axle_offset_m=self._rear_axle_offset_m,
            route_in_local=route,
        )

    def _apply_weather(self, preset: str) -> None:
        weather = getattr(self._carla.WeatherParameters, preset, None)
        if weather is None:
            raise ValueError(f"unknown CARLA weather preset {preset!r}")
        self._world.set_weather(weather)

    def _spawn_ego(self) -> Any:
        blueprints = self._world.get_blueprint_library()
        matches = blueprints.filter(self.config.ego_blueprint)
        if not matches:
            raise ValueError(
                f"no blueprint matches {self.config.ego_blueprint!r} on this CARLA build"
            )
        blueprint = matches[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "hero")

        spawn_points = self._map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError(f"map {self.scenario.map_name!r} has no spawn points")
        index = self.scenario.spawn_point_index
        if index is None:
            index = self._rng.randrange(len(spawn_points))
        transform = spawn_points[index % len(spawn_points)]

        self._ego = self._world.spawn_actor(blueprint, transform)
        # Let the actor settle before anything reads its transform.
        self._world.tick()
        return transform

    def _resolve_rear_axle_offset(self) -> float:
        if self.config.rear_axle_offset_m is not None:
            logger.info("rear axle offset: %.3f m (from config)", self.config.rear_axle_offset_m)
            return self.config.rear_axle_offset_m

        offset = self._derive_rear_axle_offset()
        logger.info(
            "rear axle offset: %.3f m (derived from wheel physics; override with "
            "RuntimeConfig.rear_axle_offset_m if the geometry looks wrong)",
            offset,
        )
        return offset

    def _derive_rear_axle_offset(self) -> float:
        """Longitudinal distance from the actor origin back to the rear axle.

        CARLA 0.9.x reports ``WheelPhysicsControl.position`` in **world
        centimetres**, not vehicle-local metres. We therefore convert the wheel
        positions into the actor frame explicitly rather than trusting them to
        already be relative.
        """
        try:
            physics = self._ego.get_physics_control()
            wheels = list(physics.wheels)
        except (AttributeError, RuntimeError):  # pragma: no cover - build dependent
            wheels = []

        if len(wheels) < 4:
            fallback = -0.5 * self._ego.bounding_box.extent.x
            logger.warning(
                "wheel physics unavailable; falling back to half the bounding box "
                "(%.3f m). Set RuntimeConfig.rear_axle_offset_m for accuracy.",
                fallback,
            )
            return float(fallback)

        actor_pose = self._actor_pose()
        world_to_actor = actor_pose.inverse()
        rear_positions = []
        for wheel in wheels[2:4]:  # CARLA orders wheels FL, FR, RL, RR
            position_cm = wheel.position
            world_point = carla_vector_to_local(
                position_cm.x / 100.0, position_cm.y / 100.0, position_cm.z / 100.0
            )
            rear_positions.append(world_to_actor.transform_points(world_point)[0])

        return float(np.mean([p[0] for p in rear_positions]))

    def _spawn_cameras(self) -> list[AvailableCamera]:
        carla = self._carla
        blueprints = self._world.get_blueprint_library()
        cameras: list[AvailableCamera] = []

        for cam in self.config.cameras:
            blueprint = blueprints.find("sensor.camera.rgb")
            blueprint.set_attribute("image_size_x", str(cam.width))
            blueprint.set_attribute("image_size_y", str(cam.height))
            blueprint.set_attribute("fov", str(cam.fov_deg))
            # One capture per simulator tick (0 = every tick in synchronous
            # mode; equal to fixed_delta_s, float accumulation can skip one,
            # and the tick then waits out its whole deadline for a frame).
            blueprint.set_attribute("sensor_tick", "0.0")

            transform = carla.Transform(
                carla.Location(x=cam.x, y=cam.y, z=cam.z),
                carla.Rotation(pitch=cam.pitch_deg, yaw=cam.yaw_deg, roll=cam.roll_deg),
            )
            sensor = self._world.spawn_actor(blueprint, transform, attach_to=self._ego)

            frames: queue.Queue = queue.Queue()
            self._frame_queues[cam.logical_id] = frames
            sensor.listen(self._make_camera_callback(cam, frames))
            self._sensors.append(sensor)

            cameras.append(
                available_camera(
                    logical_id=cam.logical_id,
                    width=cam.width,
                    height=cam.height,
                    horizontal_fov_deg=cam.fov_deg,
                    pose_in_rig=camera_pose_in_rig(
                        cam.x,
                        cam.y,
                        cam.z,
                        cam.pitch_deg,
                        cam.yaw_deg,
                        cam.roll_deg,
                        self._rear_axle_offset_m,
                    ),
                )
            )
        return cameras

    def _make_camera_callback(self, cam: CameraConfig, frames: queue.Queue):
        """Queue the raw frame; :meth:`_drain_captures` takes this tick's and encodes it.

        The sensor fires on every tick, and every tick's newest frame is
        encoded, though only a policy step's last tick is submitted. Copying the
        buffer is unavoidable (it is only valid for the duration of the
        callback).
        """
        epoch = self.config.epoch_offset_us

        def callback(image: Any) -> None:
            try:
                frames.put(
                    _RawFrame(
                        frame=int(image.frame),
                        timestamp_us=seconds_to_us(image.timestamp, epoch),
                        width=image.width,
                        height=image.height,
                        bgra=bytes(image.raw_data),
                    )
                )
            except Exception:  # pragma: no cover - sensor thread must not die
                logger.exception("failed to receive a frame from %s", cam.logical_id)
                self._events.encode_failures += 1

        return callback

    def _spawn_lidars(self) -> None:
        carla = self._carla
        blueprints = self._world.get_blueprint_library()
        for lidar in self.config.lidars:
            blueprint = blueprints.find("sensor.lidar.ray_cast")
            blueprint.set_attribute("channels", str(lidar.channels))
            blueprint.set_attribute("range", str(lidar.range_m))
            blueprint.set_attribute("points_per_second", str(lidar.points_per_second))
            # One full revolution per tick; see LidarConfig for why.
            blueprint.set_attribute("rotation_frequency", str(1.0 / self.config.fixed_delta_s))
            blueprint.set_attribute("upper_fov", str(lidar.upper_fov_deg))
            blueprint.set_attribute("lower_fov", str(lidar.lower_fov_deg))
            # 0 = every tick in synchronous mode. Equal to fixed_delta_s, float
            # accumulation can skip a tick, and the next sweep then spans two
            # revolutions.
            blueprint.set_attribute("sensor_tick", "0.0")
            if blueprint.has_attribute("dropoff_general_rate"):
                blueprint.set_attribute("dropoff_general_rate", str(lidar.dropoff_general_rate))

            transform = carla.Transform(
                carla.Location(x=lidar.x, y=lidar.y, z=lidar.z),
                carla.Rotation(pitch=lidar.pitch_deg, yaw=lidar.yaw_deg, roll=lidar.roll_deg),
            )
            sensor = self._world.spawn_actor(blueprint, transform, attach_to=self._ego)
            sweeps: queue.Queue = queue.Queue()
            self._sweep_queues[lidar.logical_id] = sweeps
            self._lidar_poses[lidar.logical_id] = sensor_pose_in_rig(
                lidar.x,
                lidar.y,
                lidar.z,
                lidar.pitch_deg,
                lidar.yaw_deg,
                lidar.roll_deg,
                self._rear_axle_offset_m,
            )
            sensor.listen(self._make_lidar_callback(lidar, sweeps))
            self._sensors.append(sensor)

    def _make_lidar_callback(self, lidar: LidarConfig, sweeps: queue.Queue):
        """Queue the raw buffer; the rig-frame conversion runs only for a sweep sent."""
        epoch = self.config.epoch_offset_us

        def callback(measurement: Any) -> None:
            try:
                points = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
                sweeps.put(
                    _RawSweep(
                        frame=int(measurement.frame),
                        timestamp_us=seconds_to_us(measurement.timestamp, epoch),
                        points_xyzi=points.copy(),
                    )
                )
            except Exception:  # pragma: no cover - sensor thread must not die
                logger.exception("failed to receive a sweep from %s", lidar.logical_id)

        return callback

    def _spawn_collision_sensors(self) -> None:
        blueprints = self._world.get_blueprint_library()
        carla = self._carla
        origin = carla.Transform()

        collision = self._world.spawn_actor(
            blueprints.find("sensor.other.collision"), origin, attach_to=self._ego
        )
        collision.listen(lambda _event: self._record_event("collision"))
        self._sensors.append(collision)

        lane = self._world.spawn_actor(
            blueprints.find("sensor.other.lane_invasion"), origin, attach_to=self._ego
        )
        lane.listen(lambda _event: self._record_event("lane_invasion"))
        self._sensors.append(lane)

    def _record_event(self, kind: str) -> None:
        if kind == "collision":
            self._events.collisions += 1
        else:
            self._events.lane_invasions += 1

    def _spawn_background_traffic(self) -> None:
        if self.scenario.num_background_vehicles <= 0:
            return
        blueprints = self._world.get_blueprint_library().filter("vehicle.*")
        spawn_points = list(self._map.get_spawn_points())
        self._rng.shuffle(spawn_points)

        spawned = 0
        for transform in spawn_points:
            if spawned >= self.scenario.num_background_vehicles:
                break
            blueprint = blueprints[self._rng.randrange(len(blueprints))]
            actor = self._world.try_spawn_actor(blueprint, transform)
            if actor is None:
                continue
            actor.set_autopilot(True, self.config.traffic_manager_port)
            self._background.append(actor)
            spawned += 1
        logger.info("spawned %d background vehicles", spawned)

    def _build_route(self, spawn_transform: Any) -> np.ndarray:
        """Follow lanes forward from the spawn point to make a driveable route.

        alpasim gets its route from the recording; there is no recording here, so
        the route is generated by walking the lane graph with seeded choices at
        junctions. Deterministic for a given ``RuntimeConfig.seed``.
        """
        step = max(0.5, self.config.route_resolution_m)
        # Long enough that a full rollout at a plausible speed stays on it.
        target_length_m = max(200.0, self.config.max_steps * self.config.policy_timestep_s * 20.0)

        waypoint = self._map.get_waypoint(spawn_transform.location, project_to_road=True)
        points = [waypoint_to_local(waypoint)]
        self._route_lane_ids = [carla_lane_key(waypoint)]
        travelled = 0.0
        while travelled < target_length_m:
            previous = waypoint
            options = waypoint.next(step)
            if not options:
                break
            waypoint = (
                options[self._rng.randrange(len(options))] if len(options) > 1 else options[0]
            )
            points.append(waypoint_to_local(waypoint))
            # A connector shorter than the step can lie between two samples;
            # look in between so it still counts as on the route.
            for fraction in (0.25, 0.5, 0.75, 1.0):
                probe = (
                    self._on_the_way(
                        previous.next(step * fraction), waypoint, step * (1 - fraction)
                    )
                    if fraction < 1.0
                    else waypoint
                )
                if probe is None:
                    continue
                lane_id = carla_lane_key(probe)
                if lane_id != self._route_lane_ids[-1]:
                    self._route_lane_ids.append(lane_id)
            travelled += step

        if len(points) < 2:
            raise RuntimeError(
                "could not build a route from the spawn point; the map may lack "
                "connected lanes at that location"
            )
        return np.stack(points)

    @staticmethod
    def _on_the_way(options: list, chosen: Any, remaining_m: float) -> Any:
        """The option that ``remaining_m`` further on reaches ``chosen``, if any.

        Past a junction's branch point ``next`` offers every branch; only the
        one leading to the waypoint the route took is on the route.
        """
        if len(options) == 1:
            return options[0]
        target = chosen.transform.location
        for option in options:
            for onward in option.next(max(remaining_m, 0.01)):
                spot = onward.transform.location
                same_lane = carla_lane_key(onward) == carla_lane_key(chosen)
                if same_lane and math.dist((spot.x, spot.y), (target.x, target.y)) < 0.5:
                    return option
        return None

    # -- stepping ----------------------------------------------------------

    def tick(self) -> WorldSnapshot:
        if self._pending_control is not None:
            self._ego.apply_control(self._to_carla_control(self._pending_control))
            self._pending_control = None

        frame_id = self._world.tick()
        deadline = time.monotonic() + _SENSOR_WAIT_S
        snapshot = self._world.get_snapshot()
        timestamp_us = seconds_to_us(
            snapshot.timestamp.elapsed_seconds, self.config.epoch_offset_us
        )
        return WorldSnapshot(
            frame_id=int(frame_id),
            timestamp_us=timestamp_us,
            ego=self._ego_state(timestamp_us),
            captures=self._drain_captures(int(frame_id), deadline),
            lidar=self._drain_sweeps(int(frame_id), deadline),
        )

    def apply_control(self, command: VehicleCommand) -> None:
        self._pending_control = command

    def _to_carla_control(self, command: VehicleCommand) -> Any:
        return self._carla.VehicleControl(
            throttle=float(np.clip(command.throttle, 0.0, 1.0)),
            steer=float(np.clip(command.steer, -1.0, 1.0)),
            brake=float(np.clip(command.brake, 0.0, 1.0)),
            hand_brake=command.hand_brake,
            reverse=command.reverse,
        )

    def _actor_pose(self) -> Pose:
        transform = self._ego.get_transform()
        location, rotation = transform.location, transform.rotation
        return carla_transform_to_pose(
            (location.x, location.y, location.z),
            (rotation.pitch, rotation.yaw, rotation.roll),
        )

    def _ego_state(self, timestamp_us: int) -> EgoState:
        transform = self._ego.get_transform()
        location, rotation = transform.location, transform.rotation
        pose = rig_pose_from_actor_transform(
            (location.x, location.y, location.z),
            (rotation.pitch, rotation.yaw, rotation.roll),
            self._rear_axle_offset_m,
        )

        velocity = self._ego.get_velocity()
        acceleration = self._ego.get_acceleration()
        angular = self._ego.get_angular_velocity()  # degrees/s in CARLA

        velocity_local = carla_vector_to_local(velocity.x, velocity.y, velocity.z)
        acceleration_local = carla_vector_to_local(acceleration.x, acceleration.y, acceleration.z)
        angular_local = carla_angular_velocity_to_local(angular.x, angular.y, angular.z)

        return EgoState(
            timestamp_us=timestamp_us,
            pose_local_to_rig=pose,
            linear_velocity_in_rig=vector_local_to_rig(velocity_local, pose),
            angular_velocity_in_rig=vector_local_to_rig(angular_local, pose),
            linear_acceleration_in_rig=vector_local_to_rig(acceleration_local, pose),
        )

    def _drain_captures(self, frame_id: int, deadline: float) -> list[CameraCapture]:
        """Take the newest frame per camera, discarding any backlog, and encode it.

        A backlog means the sensor produced more frames than this policy step
        consumes -- expected when ``sensor_tick`` is finer than the policy step,
        and also what happens if the loop falls behind. Either way, submitting a
        stale frame would silently violate the sensor-freshness invariant alpasim
        asserts (``assert_sensors_up_to_date``).
        """
        captures = []
        for logical_id, frames in self._frame_queues.items():
            newest: _RawFrame | None = _take_frame(frames, frame_id, logical_id, deadline)
            if newest is None:
                self._events.sensor_timeouts += 1
                continue
            try:
                image_bytes = encode_bgra(
                    newest.bgra,
                    newest.width,
                    newest.height,
                    self.config.image_format,
                    self.config.image_quality,
                )
            except Exception:  # pragma: no cover - must not kill the rollout
                logger.exception("failed to encode a frame from %s", logical_id)
                self._events.encode_failures += 1
                continue
            captures.append(
                CameraCapture(
                    logical_id=logical_id,
                    # A CARLA RGB capture is instantaneous, so start == end.
                    # alpasim's rolling-shutter drivers accept that; see
                    # docs/COMPATIBILITY.md.
                    frame_start_us=newest.timestamp_us,
                    frame_end_us=newest.timestamp_us,
                    image_bytes=image_bytes,
                )
            )
        return captures

    def _drain_sweeps(self, frame_id: int, deadline: float) -> list[LidarCapture]:
        """This tick's sweep per LiDAR; the rig-frame conversion is left to the reader."""
        sweeps = []
        for logical_id, pending in self._sweep_queues.items():
            newest: _RawSweep | None = _take_frame(pending, frame_id, logical_id, deadline)
            if newest is None:
                self._events.sensor_timeouts += 1
            else:
                sweeps.append(
                    LidarCapture(
                        logical_id=logical_id,
                        timestamp_us=newest.timestamp_us,
                        pose_in_rig=self._lidar_poses[logical_id],
                        points_in_sensor=newest.points_xyzi,
                    )
                )
        return sweeps

    # -- ground truth ------------------------------------------------------

    def environment(self, snapshot: WorldSnapshot) -> RendererData:
        return self._ground_truth().read(snapshot)

    def _ground_truth(self) -> CarlaGroundTruth:
        """The reader, built on first use -- it needs a world and an ego."""
        if self._ground_truth_reader is None:
            self._ground_truth_reader = CarlaGroundTruth(
                world=self._world,
                ego=self._ego,
                carla_map=self._map,
                config=self.config,
                map_name=self.scenario.map_name,
                route_lane_ids=self._route_lane_ids,
            )
        return self._ground_truth_reader

    def events(self) -> RolloutEvents:
        return self._events

    # -- teardown ----------------------------------------------------------

    def close(self) -> None:
        # Order matters, and not merely for tidiness. The traffic manager runs
        # its own thread inside the CARLA client, and while it is in
        # synchronous mode it keeps issuing commands for every vehicle
        # registered to it. Destroying those vehicles first leaves it
        # operating on actors that no longer exist, and the resulting C++
        # exception is thrown on *its* thread, where no Python `except` can
        # reach it -- the process dies with SIGABRT and
        #
        #     terminate called after throwing an instance of 'std::runtime_error'
        #       what(): trying to operate on a destroyed actor
        #
        # after a rollout that had already completed successfully. Standing
        # the traffic manager down first makes the whole teardown ordinary.
        if self._traffic_manager is not None:
            try:
                self._traffic_manager.set_synchronous_mode(False)
            except RuntimeError:  # pragma: no cover
                logger.debug("traffic manager teardown failed", exc_info=True)

        for sensor in self._sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:  # pragma: no cover - actor may already be gone
                logger.debug("sensor teardown failed", exc_info=True)
        self._sensors.clear()

        # Destroy in one batch where the client supports it: a single
        # round trip closes the window in which the server holds a
        # partially torn-down scene.
        actors = [actor for actor in [*self._background, self._ego] if actor is not None]
        if actors and self._client is not None:
            try:
                import carla

                self._client.apply_batch_sync(
                    [carla.command.DestroyActor(actor) for actor in actors], True
                )
                actors = []
            except (RuntimeError, ImportError, AttributeError):  # pragma: no cover
                logger.debug("batch actor teardown failed; falling back", exc_info=True)

        for actor in actors:
            try:
                actor.destroy()
            except RuntimeError:  # pragma: no cover
                logger.debug("actor teardown failed", exc_info=True)
        self._background.clear()
        self._ego = None

        # Give the port back, or the next run waits four seconds for it.
        #
        # The server registers a traffic manager against a port and keeps that
        # registration after the client that made it exits. A later process
        # asking for the same port therefore tries to reach a manager that is
        # no longer there, waits for the attempt to time out, and only then
        # creates a new one. Measured against this server: 0.05 s for the
        # first process to claim a port, 4.05 s for every process after it,
        # and 0.05 s throughout once `shut_down` is called -- on a 200-step
        # rollout that is a ninth of the whole run, paid every time because
        # the default port never changes.
        #
        # After the vehicles are gone rather than before, so that standing the
        # manager down keeps the ordering that stops it operating on destroyed
        # actors.
        if self._traffic_manager is not None:
            try:
                self._traffic_manager.shut_down()
            except (RuntimeError, AttributeError):  # pragma: no cover
                logger.debug("traffic manager shutdown failed", exc_info=True)
            self._traffic_manager = None

        # Leaving the server in synchronous mode would hang the next client.
        if self._world is not None and self._original_settings is not None:
            self._world.apply_settings(self._original_settings)
