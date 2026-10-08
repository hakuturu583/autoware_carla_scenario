# SPDX-License-Identifier: Apache-2.0
"""LiDAR sweeps, from the simulator's frame to a policy.

A sweep rides in ``RendererData`` because the egodriver contract has nowhere
else to put it, so the things worth pinning are the ones a reader cannot see
from the proto: which frame the numbers are in, and that a sweep survives the
trip byte for byte.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from carla_driver_interface.driver.base import BaseDriver, DriveContext, DriveResult
from carla_driver_interface.driver.policies import RouteFollowerPolicy
from carla_driver_interface.driver.server import serving
from carla_driver_interface.fakes import FakeWorld
from carla_driver_interface.geometry import Pose
from carla_driver_interface.grpc_api import ImageFormat, LidarSweep, RendererData
from carla_driver_interface.grpc_api.extension import (
    pack_lidar_sweep,
    pack_renderer_data,
    unpack_lidar_points,
    unpack_renderer_data,
)
from carla_driver_interface.runtime import CarlaRuntime, RuntimeConfig, ScenarioSpec
from carla_driver_interface.runtime.config import LidarConfig
from carla_driver_interface.runtime.conversions import lidar_points_to_rig, sensor_pose_in_rig

# ---------------------------------------------------------------------------
# LiDAR frames and codec
# ---------------------------------------------------------------------------


def test_a_level_mount_mirrors_y_and_lifts_by_the_mount_height():
    pose = sensor_pose_in_rig(0.5, 0.0, 2.0, 0.0, 0.0, 0.0, rear_axle_offset_m=-1.5)
    raw = np.array([[10.0, 3.0, -1.0, 0.25]], dtype=np.float32)  # CARLA: y right

    rig = lidar_points_to_rig(raw, pose)

    # x shifts by the mount and the actor->rig offset; y flips; z lifts.
    np.testing.assert_allclose(rig[0], [10.0 + 0.5 + 1.5, -3.0, 1.0, 0.25], atol=1e-6)


def test_a_yawed_mount_rotates_the_sweep_the_way_carla_would():
    # CARLA yaw +90 degrees turns the sensor to the vehicle's right, so a point
    # straight ahead of the sensor lies on the rig's right: -y.
    pose = sensor_pose_in_rig(0.0, 0.0, 0.0, 0.0, 90.0, 0.0, rear_axle_offset_m=0.0)
    rig = lidar_points_to_rig(np.array([[5.0, 0.0, 0.0, 1.0]], dtype=np.float32), pose)
    np.testing.assert_allclose(rig[0, :3], [0.0, -5.0, 0.0], atol=1e-5)


def test_a_sweep_round_trips_through_the_renderer_payload():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(1000, 4)).astype(np.float32)
    sweep = pack_lidar_sweep("lidar_top", 123, points, Pose.identity().to_proto())

    restored = unpack_renderer_data(pack_renderer_data(RendererData(lidar=[sweep])))

    assert restored is not None
    out = unpack_lidar_points(restored.lidar[0])
    np.testing.assert_array_equal(out, points)
    assert out.flags.writeable


def test_a_truncated_sweep_is_refused_rather_than_read_short():
    sweep = pack_lidar_sweep("lidar_top", 0, np.zeros((10, 4)), Pose.identity().to_proto())
    broken = LidarSweep()
    broken.CopyFrom(sweep)
    broken.points_xyzi = sweep.points_xyzi[:-4]
    with pytest.raises(ValueError, match="declares 10 points"):
        unpack_lidar_points(broken)


def test_packing_refuses_the_wrong_number_of_columns():
    with pytest.raises(ValueError, match=r"\[N, 4\]"):
        pack_lidar_sweep("lidar_top", 0, np.zeros((3, 5)), Pose.identity().to_proto())


def test_lidar_config_rejects_an_inverted_field_of_view():
    with pytest.raises(ValueError, match="lower_fov_deg"):
        LidarConfig(upper_fov_deg=-30.0, lower_fov_deg=10.0)


def test_lidar_logical_ids_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        RuntimeConfig(lidars=[LidarConfig(), LidarConfig()])


def test_the_fake_sweep_lands_on_the_ground_and_within_range():
    lidar = LidarConfig(channels=16, range_m=30.0, z=1.8)
    world = FakeWorld(RuntimeConfig(lidars=[lidar]))
    world.setup()
    (sweep,) = world.tick().lidar

    assert sweep.points_xyzi.dtype == np.float32
    assert len(sweep.points_xyzi) > 0
    np.testing.assert_allclose(sweep.points_xyzi[:, 2], 0.0, atol=1e-4)
    horizontal = np.linalg.norm(sweep.points_xyzi[:, :2], axis=1)
    assert horizontal.max() <= 30.0
    # A ring all the way round, not a slice: the sweep is a full revolution.
    azimuth = np.degrees(np.arctan2(sweep.points_xyzi[:, 1], sweep.points_xyzi[:, 0]))
    assert azimuth.min() < -170.0 and azimuth.max() > 170.0


def test_lidar_points_refuses_to_guess_between_several_sensors():
    sweeps = [
        pack_lidar_sweep(name, 0, np.zeros((1, 4)), Pose.identity().to_proto())
        for name in ("lidar_top", "lidar_front")
    ]
    ctx = DriveContext(
        session=None,  # type: ignore[arg-type]
        time_now_us=0,
        time_query_us=0,
        renderer_data=RendererData(lidar=sweeps),
    )
    with pytest.raises(ValueError, match="name one"):
        ctx.lidar_points()
    assert ctx.lidar_points("lidar_front").shape == (1, 4)
    assert ctx.lidar_points("missing") is None


# ---------------------------------------------------------------------------
# End to end, over gRPC
# ---------------------------------------------------------------------------


class _Recorder(BaseDriver):
    """Follows the route, and keeps what arrived for the test to inspect."""

    name = "recorder"

    def __init__(self) -> None:
        self._follower = RouteFollowerPolicy()
        self.sweep_sizes: list[int] = []

    def drive(self, ctx: DriveContext) -> DriveResult:
        points = ctx.lidar_points()
        self.sweep_sizes.append(0 if points is None else len(points))
        return self._follower.drive(ctx)


def test_sweeps_reach_the_policy_over_grpc():
    policy = _Recorder()
    scenario = ScenarioSpec(map_name="FakeTown", name="straight_then_curve")
    with serving(policy, port=0, host="127.0.0.1") as port:
        base = RuntimeConfig(
            driver_address=f"127.0.0.1:{port}",
            max_steps=30,
            image_format=ImageFormat.JPEG,
            lidars=[LidarConfig(channels=16)],
        )
        cfg = replace(base, cameras=[replace(base.cameras[0], width=32, height=24)])
        outcome = CarlaRuntime(FakeWorld(cfg, scenario), cfg, scenario).run_rollout()

    assert outcome.rollout_return.error == ""
    assert policy.sweep_sizes and min(policy.sweep_sizes) > 0


def test_nothing_extra_is_sent_unless_asked_for():
    world = FakeWorld(RuntimeConfig())
    world.setup()
    snapshot = world.tick()
    data = world.environment(snapshot)
    assert snapshot.lidar == []
    assert len(data.traffic_lights) == 0 and data.map_id == ""


# ---------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------


def test_angular_velocity_mirrors_as_an_axial_vector():
    from carla_driver_interface.runtime.conversions import carla_angular_velocity_to_local

    # CARLA yaw grows clockwise seen from above (a right turn); in the
    # right-handed local frame a right turn is a negative yaw rate.
    np.testing.assert_allclose(
        carla_angular_velocity_to_local(0.0, 0.0, 10.0), [0.0, 0.0, -math.radians(10.0)]
    )
    # Roll and pitch rates flip and keep their sign respectively, as the
    # mirror's determinant demands.
    np.testing.assert_allclose(
        carla_angular_velocity_to_local(5.0, 7.0, 0.0)[:2], np.radians([-5.0, 7.0])
    )


def test_a_rolled_mount_puts_the_fake_sweep_on_the_ground():
    # A level, centred mount sweeps a y-symmetric ring, which a lost y-mirror
    # would leave unchanged; rolled and offset, the same mistake tilts the
    # ground out of z = 0.
    lidar = LidarConfig(channels=16, range_m=30.0, y=0.4, z=1.8, roll_deg=8.0, yaw_deg=20.0)
    world = FakeWorld(RuntimeConfig(lidars=[lidar]))
    world.setup()
    (sweep,) = world.tick().lidar
    np.testing.assert_allclose(sweep.points_xyzi[:, 2], 0.0, atol=1e-3)


def test_take_frame_discards_backlog_skips_on_timeout_and_accepts_newer():
    import queue
    import time

    from carla_driver_interface.runtime import carla_world

    pending: queue.Queue = queue.Queue()
    for frame in (3, 4, 5):
        pending.put(SimpleNamespace(frame=frame))
    assert carla_world._take_frame(pending, 5, "cam", time.monotonic() + 0.5).frame == 5
    assert pending.empty()

    pending.put(SimpleNamespace(frame=6))
    started = time.monotonic()
    assert carla_world._take_frame(pending, 7, "cam", started + 0.05) is None
    assert time.monotonic() - started < 0.5

    pending.put(SimpleNamespace(frame=9))
    assert carla_world._take_frame(pending, 8, "cam", time.monotonic() + 0.05).frame == 9
