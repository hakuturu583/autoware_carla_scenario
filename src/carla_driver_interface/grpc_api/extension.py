# SPDX-License-Identifier: Apache-2.0
"""Codec for the extension payloads that ride inside alpasim's ``bytes`` fields.

Upstream leaves two fields free-form so implementers can carry their own data
without forking the proto:

* ``egodriver.DriveRequest.renderer_data`` -- runtime to driver
* ``egodriver.DriveResponse.DebugInfo.unstructured_debug_info`` -- driver to runtime

Each is written by one process and read by the other, so pack and unpack would
naturally end up in different modules and drift apart -- including in how they
handle a payload they cannot parse. Keeping both halves here makes the pair
round-trippable in a single test and gives the tolerance policy one home.

**Tolerance policy:** unpacking never raises. These fields are free-form by
definition, so a peer -- an upstream alpasim runtime, or a driver from another
project -- may legitimately put something else there. An unparseable payload
means "no extension data", not "the rollout is broken".
"""

from __future__ import annotations

import logging

import numpy as np
from alpasim_grpc.v0.common_pb2 import Pose
from google.protobuf.message import DecodeError

from carla_driver_interface.grpc_api.driver_extension.v0.driver_extension_pb2 import (
    DriveDebugInfo,
    Lane,
    LidarSweep,
    RendererData,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LIDAR_POINT_COLUMNS",
    "lane_polylines",
    "pack_lane_polylines",
    "pack_debug_info",
    "pack_lidar_sweep",
    "pack_renderer_data",
    "unpack_debug_info",
    "unpack_lidar_points",
    "unpack_renderer_data",
]

#: ``LidarSweep.points_xyzi`` is ``[N, 4]``: x, y, z, intensity.
LIDAR_POINT_COLUMNS = 4

#: The wire byte order of every packed array (``points_xyzi``, the lane
#: polylines), stated rather than left to the host.
_WIRE_DTYPE = np.dtype("<f4")


def pack_renderer_data(data: RendererData) -> bytes:
    """Serialize for ``DriveRequest.renderer_data``."""
    return data.SerializeToString()


def unpack_renderer_data(payload: bytes) -> RendererData | None:
    """Parse ``DriveRequest.renderer_data``; ``None`` if absent or foreign."""
    return _unpack(payload, RendererData, "renderer_data")


def pack_debug_info(debug: DriveDebugInfo) -> bytes:
    """Serialize for ``DriveResponse.DebugInfo.unstructured_debug_info``."""
    return debug.SerializeToString()


def unpack_debug_info(payload: bytes) -> DriveDebugInfo | None:
    """Parse the driver's debug payload; ``None`` if absent or foreign."""
    return _unpack(payload, DriveDebugInfo, "unstructured_debug_info")


def _unpack(payload: bytes, message_type: type, field: str):
    if not payload:
        return None
    message = message_type()
    try:
        message.ParseFromString(payload)
    except (DecodeError, UnicodeDecodeError):
        logger.debug("%s is not a %s; ignoring", field, message_type.__name__)
        return None
    return message


def pack_lidar_sweep(
    logical_id: str,
    timestamp_us: int,
    points_xyzi_in_rig: np.ndarray,
    rig_to_lidar: Pose,
) -> LidarSweep:
    """Build one ``LidarSweep`` from ``[N, 4]`` rig-frame points."""
    points = np.ascontiguousarray(points_xyzi_in_rig, dtype=_WIRE_DTYPE)
    if points.ndim != 2 or points.shape[1] != LIDAR_POINT_COLUMNS:
        raise ValueError(
            f"LiDAR points must be [N, {LIDAR_POINT_COLUMNS}] (x, y, z, intensity); "
            f"got shape {points.shape}"
        )
    return LidarSweep(
        logical_id=logical_id,
        timestamp_us=int(timestamp_us),
        rig_to_lidar=rig_to_lidar,
        num_points=int(points.shape[0]),
        points_xyzi=points.tobytes(),
    )


def unpack_lidar_points(sweep: LidarSweep) -> np.ndarray:
    """The sweep's points as a writable ``[N, 4]`` float32 array, rig frame.

    Raises rather than truncating when ``num_points`` and the payload disagree:
    unlike the outer ``renderer_data`` bytes, a sweep that parsed as one is ours,
    and a short buffer means it was corrupted, not that it belongs to a peer.
    """
    expected = int(sweep.num_points) * LIDAR_POINT_COLUMNS * _WIRE_DTYPE.itemsize
    if len(sweep.points_xyzi) != expected:
        raise ValueError(
            f"LiDAR sweep {sweep.logical_id!r} declares {sweep.num_points} points "
            f"({expected} bytes) but carries {len(sweep.points_xyzi)} bytes"
        )
    flat = np.frombuffer(sweep.points_xyzi, dtype=_WIRE_DTYPE)
    return flat.reshape(-1, LIDAR_POINT_COLUMNS).astype(np.float32, copy=True)


def pack_lane_polylines(
    centerline: np.ndarray, left_boundary: np.ndarray, right_boundary: np.ndarray
) -> dict:
    """``Lane`` field values for three ``[N, 3]`` rig-frame polylines of one length."""
    arrays = [
        np.ascontiguousarray(a, dtype=_WIRE_DTYPE)
        for a in (centerline, left_boundary, right_boundary)
    ]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1 or len(arrays[0].shape) != 2 or arrays[0].shape[1] != 3:
        raise ValueError(f"lane polylines must share one [N, 3] shape; got {sorted(shapes)}")
    return {
        "num_points": int(arrays[0].shape[0]),
        "centerline_xyz": arrays[0].tobytes(),
        "left_boundary_xyz": arrays[1].tobytes(),
        "right_boundary_xyz": arrays[2].tobytes(),
    }


def lane_polylines(lane: Lane) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(centerline, left_boundary, right_boundary)``, each ``[N, 3]`` float64.

    Raises when a polyline's size disagrees with ``num_points``, for the same
    reason :func:`unpack_lidar_points` does.
    """
    expected = int(lane.num_points) * 3 * _WIRE_DTYPE.itemsize
    out = []
    for name in ("centerline_xyz", "left_boundary_xyz", "right_boundary_xyz"):
        payload = getattr(lane, name)
        if len(payload) != expected:
            raise ValueError(
                f"lane {lane.lane_id!r}: {name} carries {len(payload)} bytes, "
                f"{lane.num_points} points need {expected}"
            )
        out.append(np.frombuffer(payload, dtype=_WIRE_DTYPE).reshape(-1, 3).astype(np.float64))
    return out[0], out[1], out[2]
