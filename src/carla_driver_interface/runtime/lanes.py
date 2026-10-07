# SPDX-License-Identifier: Apache-2.0
"""The drivable lanes around the ego, as ``driver_extension.v0.Lane`` messages.

Split in two, like the rest of the runtime:

* :class:`LaneGeometry` and :func:`lanes_in_rig` are simulator-free. They hold
  a lane in the ``local`` frame and crop and re-express it around the ego, and
  both the fake world and the CARLA reader go through them -- so what CI checks
  about cropping and frames is what runs against a real map.
* :func:`carla_lane_geometries` reads a ``carla.Map`` into that form. It takes
  CARLA objects but never imports ``carla``, the same arrangement as
  :mod:`carla_driver_interface.runtime.ground_truth`.

**Boundaries.** CARLA has no boundary polylines; a waypoint carries a centre
and a width. The boundaries are reconstructed as the centre offset by half the
width along the waypoint's right vector, which is exact for CARLA's own lanes
because that is how OpenDRIVE defines them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from carla_driver_interface.geometry import Pose
from carla_driver_interface.grpc_api import Lane, LaneMarkingType, TrafficLightState
from carla_driver_interface.grpc_api.extension import pack_lane_polylines
from carla_driver_interface.runtime.conversions import carla_vector_to_local

__all__ = [
    "LaneGeometry",
    "carla_lane_geometries",
    "carla_lane_key",
    "lane_marking_type",
    "lanes_in_rig",
    "route_lane_order",
]

_MARKINGS = {
    "NONE": LaneMarkingType.LANE_MARKING_TYPE_NONE,
    "Solid": LaneMarkingType.LANE_MARKING_TYPE_SOLID,
    "Broken": LaneMarkingType.LANE_MARKING_TYPE_BROKEN,
    "SolidSolid": LaneMarkingType.LANE_MARKING_TYPE_SOLID_SOLID,
    "SolidBroken": LaneMarkingType.LANE_MARKING_TYPE_SOLID_BROKEN,
    "BrokenSolid": LaneMarkingType.LANE_MARKING_TYPE_BROKEN_SOLID,
    "BrokenBroken": LaneMarkingType.LANE_MARKING_TYPE_BROKEN_BROKEN,
    "BottsDots": LaneMarkingType.LANE_MARKING_TYPE_BOTTS_DOTS,
    "Grass": LaneMarkingType.LANE_MARKING_TYPE_GRASS,
    "Curb": LaneMarkingType.LANE_MARKING_TYPE_CURB,
    "Other": LaneMarkingType.LANE_MARKING_TYPE_OTHER,
}


def lane_marking_type(marking: Any) -> LaneMarkingType:
    """A ``carla.LaneMarking`` (or ``None``) -> the wire enum.

    Matched on the enum's name rather than its integer, because the Python
    binding's integer values are an implementation detail of the build.
    """
    if marking is None:
        return LaneMarkingType.LANE_MARKING_TYPE_UNKNOWN
    name = str(getattr(marking, "type", marking)).rsplit(".", 1)[-1]
    return _MARKINGS.get(name, LaneMarkingType.LANE_MARKING_TYPE_UNKNOWN)


@dataclass(frozen=True, eq=False)
class LaneGeometry:
    """One lane in the ``local`` frame, sampled in its driving direction."""

    lane_id: str
    #: ``[N, 3]`` each, point ``i`` of the three describing one cross-section.
    centerline: np.ndarray
    left_boundary: np.ndarray
    right_boundary: np.ndarray
    left_marking: LaneMarkingType = LaneMarkingType.LANE_MARKING_TYPE_UNKNOWN
    right_marking: LaneMarkingType = LaneMarkingType.LANE_MARKING_TYPE_UNKNOWN
    is_junction: bool = False
    #: m/s; 0 when unknown.
    speed_limit_mps: float = 0.0
    #: ``(centre, radius)`` of the centreline's bounding circle, derived.
    bounds: tuple[np.ndarray, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        shapes = {
            np.shape(self.centerline),
            np.shape(self.left_boundary),
            np.shape(self.right_boundary),
        }
        if len(shapes) != 1 or len(next(iter(shapes))) != 2 or next(iter(shapes))[1] != 3:
            raise ValueError(f"lane {self.lane_id!r}: polylines must share one [N, 3] shape")
        # A bounding circle, so a lane far beyond the horizon is rejected
        # without measuring each of its stations every step.
        xy = np.asarray(self.centerline, dtype=np.float64)[:, :2]
        centre = xy.mean(axis=0) if len(xy) else np.zeros(2)
        radius = float(np.max(np.linalg.norm(xy - centre, axis=1))) if len(xy) else 0.0
        object.__setattr__(self, "bounds", (centre, radius))


def lanes_in_rig(
    lanes: Iterable[LaneGeometry],
    pose_local_to_rig: Pose,
    horizon_m: float,
    route_order: Mapping[str, int] | None = None,
    traffic_lights: Mapping[str, TrafficLightState] | None = None,
) -> list[Lane]:
    """The lanes within ``horizon_m`` of the ego, re-expressed in the rig frame.

    A lane leaving the horizon is cropped to the contiguous run of stations
    nearest the ego, not to every station that happens to fall inside: a lane
    that bends out of range and back in would otherwise come back as one
    polyline with a jump across the gap.

    The result is ordered by distance to the ego, nearest first, so a consumer
    that keeps only the first ``k`` keeps the ones that matter.
    """
    route_order = route_order or {}
    traffic_lights = traffic_lights or {}
    to_rig = pose_local_to_rig.inverse()
    ego_xy = pose_local_to_rig.position[:2]

    found: list[tuple[float, Lane]] = []
    for lane in lanes:
        centre, radius = lane.bounds
        if len(lane.centerline) < 2 or np.linalg.norm(centre - ego_xy) - radius > horizon_m:
            continue
        distance = np.linalg.norm(lane.centerline[:, :2] - ego_xy, axis=1)
        inside = distance <= horizon_m
        if not inside.any():
            continue
        nearest = int(np.argmin(distance))
        start, stop = _run_around(inside, nearest)
        if stop - start < 2:
            continue
        window = slice(start, stop)

        def rig(points: np.ndarray, window: slice = window) -> np.ndarray:
            return to_rig.transform_points(points[window])

        found.append(
            (
                float(distance[nearest]),
                Lane(
                    lane_id=lane.lane_id,
                    **pack_lane_polylines(
                        rig(lane.centerline), rig(lane.left_boundary), rig(lane.right_boundary)
                    ),
                    left_marking=lane.left_marking,
                    right_marking=lane.right_marking,
                    traffic_light=traffic_lights.get(
                        lane.lane_id, TrafficLightState.TRAFFIC_LIGHT_STATE_NONE
                    ),
                    speed_limit_mps=float(lane.speed_limit_mps),
                    route_index=int(route_order.get(lane.lane_id, -1)),
                    is_junction=bool(lane.is_junction),
                ),
            )
        )
    found.sort(key=lambda item: item[0])
    return [lane for _, lane in found]


def _run_around(inside: np.ndarray, index: int) -> tuple[int, int]:
    """The ``[start, stop)`` run of ``True`` containing ``index`` (which is ``True``)."""
    start = index
    while start > 0 and inside[start - 1]:
        start -= 1
    stop = index + 1
    while stop < len(inside) and inside[stop]:
        stop += 1
    return start, stop


def route_lane_order(lane_ids: Iterable[str]) -> dict[str, int]:
    """Lane ids in the order a route visits them -> their position along it.

    A lane visited twice keeps its first position: the route is a walk, and a
    loop back onto an earlier lane does not move where that lane first began.
    """
    order: dict[str, int] = {}
    for lane_id in lane_ids:
        if lane_id not in order:
            order[lane_id] = len(order)
    return order


def carla_lane_key(waypoint: Any) -> str:
    """The ``Lane.lane_id`` of a CARLA waypoint."""
    return f"{waypoint.road_id}:{waypoint.section_id}:{waypoint.lane_id}"


def carla_lane_geometries(
    carla_map: Any,
    resolution_m: float,
    speed_limit_for: Callable[[Any], float],
) -> list[LaneGeometry]:
    """Every driving lane of a ``carla.Map``, in the ``local`` frame.

    ``generate_waypoints`` samples the whole map once; the samples are grouped
    by lane and ordered along it by ``s``. A lane whose OpenDRIVE id is
    positive runs against increasing ``s``, so those are reversed to put every
    polyline in its driving direction -- which is what makes "left" mean left.
    """
    groups: dict[str, list[Any]] = {}
    for waypoint in carla_map.generate_waypoints(resolution_m):
        groups.setdefault(carla_lane_key(waypoint), []).append(waypoint)

    lanes = []
    for lane_id, waypoints in groups.items():
        waypoints.sort(key=lambda wp: wp.s)
        if waypoints[0].lane_id > 0:
            waypoints.reverse()
        # generate_waypoints samples every `resolution_m`, so a lane stops up to
        # that short of its end and a section shorter than it is one point.
        # Walking on to the lane end closes the gap to the next section and
        # keeps short connectors in the graph.
        tail = getattr(waypoints[-1], "next_until_lane_end", None)
        if tail is not None:
            waypoints.extend(wp for wp in tail(resolution_m) if carla_lane_key(wp) == lane_id)
        if len(waypoints) < 2:
            continue
        centre, left, right = [], [], []
        for waypoint in waypoints:
            location = waypoint.transform.location
            right_vector = waypoint.transform.get_right_vector()
            half = 0.5 * float(waypoint.lane_width)
            c = carla_vector_to_local(location.x, location.y, location.z)
            # The right vector is in CARLA's frame; mirrored like any vector.
            r = carla_vector_to_local(right_vector.x, right_vector.y, right_vector.z)
            centre.append(c)
            left.append(c - half * r)
            right.append(c + half * r)
        first = waypoints[0]
        lanes.append(
            LaneGeometry(
                lane_id=lane_id,
                centerline=np.stack(centre),
                left_boundary=np.stack(left),
                right_boundary=np.stack(right),
                left_marking=lane_marking_type(getattr(first, "left_lane_marking", None)),
                right_marking=lane_marking_type(getattr(first, "right_lane_marking", None)),
                is_junction=bool(first.is_junction),
                speed_limit_mps=float(speed_limit_for(first)),
            )
        )
    return lanes
