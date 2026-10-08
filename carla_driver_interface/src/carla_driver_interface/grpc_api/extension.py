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
from google.protobuf.message import DecodeError

from carla_driver_interface.grpc_api._proto.common_pb2 import Pose
from carla_driver_interface.grpc_api._proto.driver_extension_pb2 import (
    DriveDebugInfo,
    LidarSweep,
    RendererData,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LIDAR_POINT_COLUMNS",
    "pack_debug_info",
    "pack_lidar_sweep",
    "pack_renderer_data",
    "unpack_debug_info",
    "unpack_lidar_points",
    "unpack_renderer_data",
]

#: ``LidarSweep.points_xyzi`` is ``[N, 4]``: x, y, z, intensity.
LIDAR_POINT_COLUMNS = 4

#: The wire byte order of the packed LiDAR points (``points_xyzi``),
#: stated rather than left to the host.
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
