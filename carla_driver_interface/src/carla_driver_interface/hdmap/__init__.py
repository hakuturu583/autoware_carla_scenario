# SPDX-License-Identifier: Apache-2.0
"""The map as files: written once by the runtime, read by drivers.

:mod:`.export` (runtime side, needs roadgen) converts a world's OpenDRIVE into
the formats drivers read; :mod:`.files` (driver side) opens one and resolves
each step's traffic lights into stop lines in that format.
"""

from carla_driver_interface.hdmap.files import MapFiles, StopLine

__all__ = ["MapFiles", "StopLine"]
