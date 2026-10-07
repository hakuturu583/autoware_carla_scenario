# SPDX-License-Identifier: Apache-2.0
"""LiDAR sweeps and the lane map, from the simulator's frame to a policy.

Both ride in ``RendererData`` because the egodriver contract has nowhere else
to put them, so the things worth pinning are the ones a reader cannot see from
the proto: which frame the numbers are in, which side "left" is, and that a
sweep survives the trip byte for byte.
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
from carla_driver_interface.fakes.fake_world import FAKE_NEIGHBOUR_LANE_ID, FAKE_ROUTE_LANE_ID
from carla_driver_interface.geometry import Pose
from carla_driver_interface.grpc_api import (
    ImageFormat,
    LaneMarkingType,
    LidarSweep,
    RendererData,
    TrafficLightState,
)
from carla_driver_interface.grpc_api.extension import (
    lane_polylines,
    pack_lidar_sweep,
    pack_renderer_data,
    unpack_lidar_points,
    unpack_renderer_data,
)
from carla_driver_interface.runtime import CarlaRuntime, RuntimeConfig, ScenarioSpec
from carla_driver_interface.runtime.config import LidarConfig
from carla_driver_interface.runtime.conversions import lidar_points_to_rig, sensor_pose_in_rig
from carla_driver_interface.runtime.lanes import (
    LaneGeometry,
    carla_lane_geometries,
    lane_marking_type,
    lanes_in_rig,
    route_lane_order,
)

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
# Lanes
# ---------------------------------------------------------------------------


def straight_lane(lane_id: str, y: float, length: float = 200.0, step: float = 2.0):
    xs = np.arange(0.0, length, step)
    centre = np.stack([xs, np.full_like(xs, y), np.zeros_like(xs)], axis=1)
    return LaneGeometry(
        lane_id=lane_id,
        centerline=centre,
        left_boundary=centre + [0.0, 1.75, 0.0],
        right_boundary=centre - [0.0, 1.75, 0.0],
    )


def test_lanes_are_cropped_to_the_horizon_and_expressed_in_the_rig_frame():
    lane = straight_lane("a", y=0.0)
    ego = Pose.from_xyz_yaw(100.0, 0.0, 0.0, math.pi / 2)  # facing +y in local

    (out,) = lanes_in_rig([lane], ego, horizon_m=20.0)
    centre, left, right = lane_polylines(out)

    # The lane runs along local +x, which is the rig's -y for this ego.
    assert np.all(np.abs(centre[:, 0]) < 1e-9)
    assert centre[:, 1].min() >= -20.0 - 1e-9 and centre[:, 1].max() <= 20.0 + 1e-9
    # Its left (local +y) is the rig's +x: ahead of the ego.
    np.testing.assert_allclose(left[:, 0], 1.75)
    np.testing.assert_allclose(right[:, 0], -1.75)


def test_a_lane_bending_out_and_back_is_cropped_to_one_contiguous_run():
    xs = np.linspace(-50.0, 50.0, 101)
    # A hairpin: out to x=+50, then back towards the ego on a parallel line.
    centre = np.concatenate(
        [
            np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], 1),
            np.stack([xs[::-1], np.full_like(xs, 5.0), np.zeros_like(xs)], 1),
        ]
    )
    lane = LaneGeometry("hairpin", centre, centre + [0, 1, 0], centre - [0, 1, 0])

    (out,) = lanes_in_rig([lane], Pose.identity(), horizon_m=20.0)
    centre_out, _, _ = lane_polylines(out)

    steps = np.linalg.norm(np.diff(centre_out[:, :2], axis=0), axis=1)
    assert steps.max() < 1.5, "the crop joined two separate runs across a gap"


def test_lanes_come_nearest_first_with_their_route_and_signal():
    lanes = [straight_lane("far", y=30.0), straight_lane("near", y=3.5), straight_lane("ego", 0.0)]
    out = lanes_in_rig(
        lanes,
        Pose.from_xyz_yaw(50.0, 0.0, 0.0, 0.0),
        horizon_m=100.0,
        route_order=route_lane_order(["ego", "ego", "far"]),
        traffic_lights={"near": TrafficLightState.TRAFFIC_LIGHT_STATE_RED},
    )
    assert [lane.lane_id for lane in out] == ["ego", "near", "far"]
    assert [lane.route_index for lane in out] == [0, -1, 1]
    assert out[1].traffic_light == TrafficLightState.TRAFFIC_LIGHT_STATE_RED
    assert out[0].traffic_light == TrafficLightState.TRAFFIC_LIGHT_STATE_NONE


def test_a_lane_entirely_outside_the_horizon_is_not_sent():
    assert lanes_in_rig([straight_lane("a", y=500.0)], Pose.identity(), horizon_m=100.0) == []


def test_lane_geometry_refuses_mismatched_polylines():
    centre = np.zeros((5, 3))
    with pytest.raises(ValueError, match="share one"):
        LaneGeometry("bad", centre, np.zeros((4, 3)), centre)


# -- the CARLA map reader, against stand-in waypoints ------------------------


def _waypoint(road, section, lane, s, x, y, yaw_deg, width=3.5, left="Solid", right="Broken"):
    yaw = math.radians(yaw_deg)
    # CARLA's right vector for a heading of yaw in its left-handed frame.
    right_vector = SimpleNamespace(x=-math.sin(yaw), y=math.cos(yaw), z=0.0)
    return SimpleNamespace(
        road_id=road,
        section_id=section,
        lane_id=lane,
        s=s,
        lane_width=width,
        is_junction=False,
        left_lane_marking=SimpleNamespace(type=f"LaneMarkingType.{left}"),
        right_lane_marking=SimpleNamespace(type=f"LaneMarkingType.{right}"),
        transform=SimpleNamespace(
            location=SimpleNamespace(x=x, y=y, z=0.0),
            get_right_vector=lambda rv=right_vector: rv,
        ),
        # Ends here unless a test says otherwise: past a lane's end, CARLA's
        # Waypoint.next returns no waypoint at all.
        next=lambda d: [],
    )


class _StandInMap:
    def __init__(self, waypoints):
        self._waypoints = waypoints

    def generate_waypoints(self, resolution):
        return list(self._waypoints)


def test_carla_lanes_run_in_their_driving_direction_with_left_on_the_left():
    # Road 1 runs along CARLA +x. Lane -1 drives with s; lane +1 against it.
    forward = [_waypoint(1, 0, -1, s, s, 1.75, 0.0) for s in (0.0, 2.0, 4.0)]
    backward = [_waypoint(1, 0, 1, s, s, -1.75, 180.0) for s in (4.0, 0.0, 2.0)]

    lanes = {
        lane.lane_id: lane
        for lane in carla_lane_geometries(_StandInMap(forward + backward), 2.0, lambda wp: 8.0)
    }

    ahead = lanes["1:0:-1"]
    np.testing.assert_allclose(ahead.centerline[:, 0], [0.0, 2.0, 4.0])
    # CARLA y=+1.75 (right of the road) is local y=-1.75; its left is towards
    # the road's centre, i.e. local +y.
    np.testing.assert_allclose(ahead.left_boundary[:, 1], 0.0, atol=1e-9)
    np.testing.assert_allclose(ahead.right_boundary[:, 1], -3.5, atol=1e-9)
    assert ahead.left_marking == LaneMarkingType.LANE_MARKING_TYPE_SOLID
    assert ahead.right_marking == LaneMarkingType.LANE_MARKING_TYPE_BROKEN
    assert ahead.speed_limit_mps == 8.0

    oncoming = lanes["1:0:1"]
    np.testing.assert_allclose(oncoming.centerline[:, 0], [4.0, 2.0, 0.0])
    np.testing.assert_allclose(oncoming.left_boundary[:, 1], 0.0, atol=1e-9)


def test_an_unknown_marking_is_reported_as_unknown_not_guessed():
    assert lane_marking_type(None) == LaneMarkingType.LANE_MARKING_TYPE_UNKNOWN
    unknown = SimpleNamespace(type="LaneMarkingType.SomethingNew")
    assert lane_marking_type(unknown) == LaneMarkingType.LANE_MARKING_TYPE_UNKNOWN


# ---------------------------------------------------------------------------
# End to end, over gRPC
# ---------------------------------------------------------------------------


class _Recorder(BaseDriver):
    """Follows the route, and keeps what arrived for the test to inspect."""

    name = "recorder"

    def __init__(self) -> None:
        self._follower = RouteFollowerPolicy()
        self.sweep_sizes: list[int] = []
        self.lane_ids: list[list[str]] = []
        self.route_lane_ahead: list[float] = []

    def drive(self, ctx: DriveContext) -> DriveResult:
        points = ctx.lidar_points()
        self.sweep_sizes.append(0 if points is None else len(points))
        lanes = ctx.lanes()
        self.lane_ids.append([lane.lane_id for lane in lanes])
        on_route = [lane for lane in lanes if lane.route_index == 0]
        if on_route:
            centre, _, _ = lane_polylines(on_route[0])
            self.route_lane_ahead.append(float(centre[:, 0].max()))
        return self._follower.drive(ctx)


def test_sweeps_and_lanes_reach_the_policy_over_grpc():
    policy = _Recorder()
    scenario = ScenarioSpec(map_name="FakeTown", name="straight_then_curve")
    with serving(policy, port=0, host="127.0.0.1") as port:
        base = RuntimeConfig(
            driver_address=f"127.0.0.1:{port}",
            max_steps=30,
            image_format=ImageFormat.JPEG,
            lidars=[LidarConfig(channels=16)],
            send_lanes=True,
            lane_horizon_m=40.0,
        )
        cfg = replace(base, cameras=[replace(base.cameras[0], width=32, height=24)])
        outcome = CarlaRuntime(FakeWorld(cfg, scenario), cfg, scenario).run_rollout()

    assert outcome.rollout_return.error == ""
    assert policy.sweep_sizes and min(policy.sweep_sizes) > 0
    assert all(ids[0] == FAKE_ROUTE_LANE_ID for ids in policy.lane_ids)
    assert all(FAKE_NEIGHBOUR_LANE_ID in ids for ids in policy.lane_ids)
    # The route lane reaches ahead of the ego, as far as the horizon allows.
    assert min(policy.route_lane_ahead) > 30.0


def test_nothing_extra_is_sent_unless_asked_for():
    world = FakeWorld(RuntimeConfig())
    world.setup()
    snapshot = world.tick()
    data = world.environment(snapshot)
    assert snapshot.lidar == []
    assert len(data.lanes) == 0


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


def _ends_at(waypoint, lane_end_x: float, beyond):
    """Give a stand-in waypoint ``next(d)`` for a lane ending at ``lane_end_x``."""

    def next_(d):
        x = waypoint.transform.location.x + d
        if x <= lane_end_x:
            return [_waypoint(1, 0, -1, x, x, 1.75, 0.0)]
        return beyond

    waypoint.next = next_
    return waypoint


def test_a_lane_runs_on_to_its_end_past_the_last_sample():
    elsewhere = _waypoint(2, 0, -1, 0.0, 3.0, 1.75, 0.0)
    first = _ends_at(_waypoint(1, 0, -1, 0.0, 0.0, 1.75, 0.0), 1.5, [elsewhere])

    (lane,) = carla_lane_geometries(_StandInMap([first]), 2.0, lambda wp: 0.0)
    # A one-sample section, kept by finding where it ends.
    np.testing.assert_allclose(lane.centerline[:, 0], [0.0, 1.5], atol=0.01)


def test_a_lane_ending_mid_road_is_read_without_walking_off_it():
    # A merge: past its end the lane has no successor at all.
    first = _ends_at(_waypoint(1, 0, -1, 0.0, 0.0, 1.75, 0.0), 0.8, [])
    (lane,) = carla_lane_geometries(_StandInMap([first]), 2.0, lambda wp: 0.0)
    np.testing.assert_allclose(lane.centerline[-1, 0], 0.8, atol=0.01)


def test_a_positive_lane_is_extended_at_its_driving_direction_end():
    # Lane +1 drives against s: samples at s = 4, 2, 0 run x = 4 -> 0, and the
    # lane goes on to x = -1.2 past the last sample.
    samples = [_waypoint(1, 0, 1, s, s, -1.75, 180.0) for s in (0.0, 2.0, 4.0)]

    def next_(d):
        x = -d
        return [_waypoint(1, 0, 1, 0.0, x, -1.75, 180.0)] if x >= -1.2 else []

    samples[0].next = next_  # the s = 0 sample is the driving-direction end
    (lane,) = carla_lane_geometries(_StandInMap(samples), 2.0, lambda wp: 0.0)
    np.testing.assert_allclose(lane.centerline[:, 0], [4.0, 2.0, 0.0, -1.2], atol=0.01)
