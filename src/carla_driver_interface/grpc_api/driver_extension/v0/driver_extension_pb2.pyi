from alpasim_grpc.v0 import common_pb2 as _common_pb2
from alpasim_grpc.v0 import egodriver_pb2 as _egodriver_pb2
from alpasim_grpc.v0 import sensorsim_pb2 as _sensorsim_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TrafficLightState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRAFFIC_LIGHT_STATE_UNKNOWN: _ClassVar[TrafficLightState]
    TRAFFIC_LIGHT_STATE_NONE: _ClassVar[TrafficLightState]
    TRAFFIC_LIGHT_STATE_RED: _ClassVar[TrafficLightState]
    TRAFFIC_LIGHT_STATE_YELLOW: _ClassVar[TrafficLightState]
    TRAFFIC_LIGHT_STATE_GREEN: _ClassVar[TrafficLightState]
    TRAFFIC_LIGHT_STATE_OFF: _ClassVar[TrafficLightState]

class LaneMarkingType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LANE_MARKING_TYPE_UNKNOWN: _ClassVar[LaneMarkingType]
    LANE_MARKING_TYPE_NONE: _ClassVar[LaneMarkingType]
    LANE_MARKING_TYPE_SOLID: _ClassVar[LaneMarkingType]
    LANE_MARKING_TYPE_BROKEN: _ClassVar[LaneMarkingType]
    LANE_MARKING_TYPE_SOLID_SOLID: _ClassVar[LaneMarkingType]
    LANE_MARKING_TYPE_SOLID_BROKEN: _ClassVar[LaneMarkingType]
    LANE_MARKING_TYPE_BROKEN_SOLID: _ClassVar[LaneMarkingType]
    LANE_MARKING_TYPE_BROKEN_BROKEN: _ClassVar[LaneMarkingType]
    LANE_MARKING_TYPE_BOTTS_DOTS: _ClassVar[LaneMarkingType]
    LANE_MARKING_TYPE_GRASS: _ClassVar[LaneMarkingType]
    LANE_MARKING_TYPE_CURB: _ClassVar[LaneMarkingType]
    LANE_MARKING_TYPE_OTHER: _ClassVar[LaneMarkingType]

class CompatLevel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COMPAT_LEVEL_UNSPECIFIED: _ClassVar[CompatLevel]
    COMPAT_LEVEL_EXACT: _ClassVar[CompatLevel]
    COMPAT_LEVEL_PARTIAL: _ClassVar[CompatLevel]
    COMPAT_LEVEL_STRUCTURAL: _ClassVar[CompatLevel]
    COMPAT_LEVEL_UNIMPLEMENTED: _ClassVar[CompatLevel]
    COMPAT_LEVEL_EXTENSION: _ClassVar[CompatLevel]
TRAFFIC_LIGHT_STATE_UNKNOWN: TrafficLightState
TRAFFIC_LIGHT_STATE_NONE: TrafficLightState
TRAFFIC_LIGHT_STATE_RED: TrafficLightState
TRAFFIC_LIGHT_STATE_YELLOW: TrafficLightState
TRAFFIC_LIGHT_STATE_GREEN: TrafficLightState
TRAFFIC_LIGHT_STATE_OFF: TrafficLightState
LANE_MARKING_TYPE_UNKNOWN: LaneMarkingType
LANE_MARKING_TYPE_NONE: LaneMarkingType
LANE_MARKING_TYPE_SOLID: LaneMarkingType
LANE_MARKING_TYPE_BROKEN: LaneMarkingType
LANE_MARKING_TYPE_SOLID_SOLID: LaneMarkingType
LANE_MARKING_TYPE_SOLID_BROKEN: LaneMarkingType
LANE_MARKING_TYPE_BROKEN_SOLID: LaneMarkingType
LANE_MARKING_TYPE_BROKEN_BROKEN: LaneMarkingType
LANE_MARKING_TYPE_BOTTS_DOTS: LaneMarkingType
LANE_MARKING_TYPE_GRASS: LaneMarkingType
LANE_MARKING_TYPE_CURB: LaneMarkingType
LANE_MARKING_TYPE_OTHER: LaneMarkingType
COMPAT_LEVEL_UNSPECIFIED: CompatLevel
COMPAT_LEVEL_EXACT: CompatLevel
COMPAT_LEVEL_PARTIAL: CompatLevel
COMPAT_LEVEL_STRUCTURAL: CompatLevel
COMPAT_LEVEL_UNIMPLEMENTED: CompatLevel
COMPAT_LEVEL_EXTENSION: CompatLevel

class Weather(_message.Message):
    __slots__ = ("cloudiness", "precipitation", "precipitation_deposits", "wind_intensity", "sun_azimuth_angle", "sun_altitude_angle", "fog_density", "wetness")
    CLOUDINESS_FIELD_NUMBER: _ClassVar[int]
    PRECIPITATION_FIELD_NUMBER: _ClassVar[int]
    PRECIPITATION_DEPOSITS_FIELD_NUMBER: _ClassVar[int]
    WIND_INTENSITY_FIELD_NUMBER: _ClassVar[int]
    SUN_AZIMUTH_ANGLE_FIELD_NUMBER: _ClassVar[int]
    SUN_ALTITUDE_ANGLE_FIELD_NUMBER: _ClassVar[int]
    FOG_DENSITY_FIELD_NUMBER: _ClassVar[int]
    WETNESS_FIELD_NUMBER: _ClassVar[int]
    cloudiness: float
    precipitation: float
    precipitation_deposits: float
    wind_intensity: float
    sun_azimuth_angle: float
    sun_altitude_angle: float
    fog_density: float
    wetness: float
    def __init__(self, cloudiness: _Optional[float] = ..., precipitation: _Optional[float] = ..., precipitation_deposits: _Optional[float] = ..., wind_intensity: _Optional[float] = ..., sun_azimuth_angle: _Optional[float] = ..., sun_altitude_angle: _Optional[float] = ..., fog_density: _Optional[float] = ..., wetness: _Optional[float] = ...) -> None: ...

class ActorState(_message.Message):
    __slots__ = ("track_id", "type_id", "pose_local_to_aabb", "aabb", "dynamic_state")
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    TYPE_ID_FIELD_NUMBER: _ClassVar[int]
    POSE_LOCAL_TO_AABB_FIELD_NUMBER: _ClassVar[int]
    AABB_FIELD_NUMBER: _ClassVar[int]
    DYNAMIC_STATE_FIELD_NUMBER: _ClassVar[int]
    track_id: str
    type_id: str
    pose_local_to_aabb: _common_pb2.Pose
    aabb: _common_pb2.AABB
    dynamic_state: _common_pb2.DynamicState
    def __init__(self, track_id: _Optional[str] = ..., type_id: _Optional[str] = ..., pose_local_to_aabb: _Optional[_Union[_common_pb2.Pose, _Mapping]] = ..., aabb: _Optional[_Union[_common_pb2.AABB, _Mapping]] = ..., dynamic_state: _Optional[_Union[_common_pb2.DynamicState, _Mapping]] = ...) -> None: ...

class RendererData(_message.Message):
    __slots__ = ("snapshot_timestamp_us", "frame_id", "map_name", "weather", "ego_traffic_light", "ego_traffic_light_distance_m", "speed_limit_mps", "actors", "lanes", "lidar")
    SNAPSHOT_TIMESTAMP_US_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    MAP_NAME_FIELD_NUMBER: _ClassVar[int]
    WEATHER_FIELD_NUMBER: _ClassVar[int]
    EGO_TRAFFIC_LIGHT_FIELD_NUMBER: _ClassVar[int]
    EGO_TRAFFIC_LIGHT_DISTANCE_M_FIELD_NUMBER: _ClassVar[int]
    SPEED_LIMIT_MPS_FIELD_NUMBER: _ClassVar[int]
    ACTORS_FIELD_NUMBER: _ClassVar[int]
    LANES_FIELD_NUMBER: _ClassVar[int]
    LIDAR_FIELD_NUMBER: _ClassVar[int]
    snapshot_timestamp_us: int
    frame_id: int
    map_name: str
    weather: Weather
    ego_traffic_light: TrafficLightState
    ego_traffic_light_distance_m: float
    speed_limit_mps: float
    actors: _containers.RepeatedCompositeFieldContainer[ActorState]
    lanes: _containers.RepeatedCompositeFieldContainer[Lane]
    lidar: _containers.RepeatedCompositeFieldContainer[LidarSweep]
    def __init__(self, snapshot_timestamp_us: _Optional[int] = ..., frame_id: _Optional[int] = ..., map_name: _Optional[str] = ..., weather: _Optional[_Union[Weather, _Mapping]] = ..., ego_traffic_light: _Optional[_Union[TrafficLightState, str]] = ..., ego_traffic_light_distance_m: _Optional[float] = ..., speed_limit_mps: _Optional[float] = ..., actors: _Optional[_Iterable[_Union[ActorState, _Mapping]]] = ..., lanes: _Optional[_Iterable[_Union[Lane, _Mapping]]] = ..., lidar: _Optional[_Iterable[_Union[LidarSweep, _Mapping]]] = ...) -> None: ...

class Lane(_message.Message):
    __slots__ = ("lane_id", "centerline", "left_boundary", "right_boundary", "left_marking", "right_marking", "traffic_light", "speed_limit_mps", "route_index", "is_junction")
    LANE_ID_FIELD_NUMBER: _ClassVar[int]
    CENTERLINE_FIELD_NUMBER: _ClassVar[int]
    LEFT_BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    RIGHT_BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    LEFT_MARKING_FIELD_NUMBER: _ClassVar[int]
    RIGHT_MARKING_FIELD_NUMBER: _ClassVar[int]
    TRAFFIC_LIGHT_FIELD_NUMBER: _ClassVar[int]
    SPEED_LIMIT_MPS_FIELD_NUMBER: _ClassVar[int]
    ROUTE_INDEX_FIELD_NUMBER: _ClassVar[int]
    IS_JUNCTION_FIELD_NUMBER: _ClassVar[int]
    lane_id: str
    centerline: _containers.RepeatedCompositeFieldContainer[_common_pb2.Vec3]
    left_boundary: _containers.RepeatedCompositeFieldContainer[_common_pb2.Vec3]
    right_boundary: _containers.RepeatedCompositeFieldContainer[_common_pb2.Vec3]
    left_marking: LaneMarkingType
    right_marking: LaneMarkingType
    traffic_light: TrafficLightState
    speed_limit_mps: float
    route_index: int
    is_junction: bool
    def __init__(self, lane_id: _Optional[str] = ..., centerline: _Optional[_Iterable[_Union[_common_pb2.Vec3, _Mapping]]] = ..., left_boundary: _Optional[_Iterable[_Union[_common_pb2.Vec3, _Mapping]]] = ..., right_boundary: _Optional[_Iterable[_Union[_common_pb2.Vec3, _Mapping]]] = ..., left_marking: _Optional[_Union[LaneMarkingType, str]] = ..., right_marking: _Optional[_Union[LaneMarkingType, str]] = ..., traffic_light: _Optional[_Union[TrafficLightState, str]] = ..., speed_limit_mps: _Optional[float] = ..., route_index: _Optional[int] = ..., is_junction: bool = ...) -> None: ...

class LidarSweep(_message.Message):
    __slots__ = ("logical_id", "timestamp_us", "rig_to_lidar", "num_points", "points_xyzi")
    LOGICAL_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_US_FIELD_NUMBER: _ClassVar[int]
    RIG_TO_LIDAR_FIELD_NUMBER: _ClassVar[int]
    NUM_POINTS_FIELD_NUMBER: _ClassVar[int]
    POINTS_XYZI_FIELD_NUMBER: _ClassVar[int]
    logical_id: str
    timestamp_us: int
    rig_to_lidar: _common_pb2.Pose
    num_points: int
    points_xyzi: bytes
    def __init__(self, logical_id: _Optional[str] = ..., timestamp_us: _Optional[int] = ..., rig_to_lidar: _Optional[_Union[_common_pb2.Pose, _Mapping]] = ..., num_points: _Optional[int] = ..., points_xyzi: _Optional[bytes] = ...) -> None: ...

class DriveDebugInfo(_message.Message):
    __slots__ = ("policy_name", "inference_seconds", "scalars")
    class ScalarsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    POLICY_NAME_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SCALARS_FIELD_NUMBER: _ClassVar[int]
    policy_name: str
    inference_seconds: float
    scalars: _containers.ScalarMap[str, float]
    def __init__(self, policy_name: _Optional[str] = ..., inference_seconds: _Optional[float] = ..., scalars: _Optional[_Mapping[str, float]] = ...) -> None: ...

class DriveSessionInfo(_message.Message):
    __slots__ = ("base", "map_name", "scenario_name", "fixed_delta_us", "policy_timestep_us", "rear_axle_offset_m", "cameras")
    BASE_FIELD_NUMBER: _ClassVar[int]
    MAP_NAME_FIELD_NUMBER: _ClassVar[int]
    SCENARIO_NAME_FIELD_NUMBER: _ClassVar[int]
    FIXED_DELTA_US_FIELD_NUMBER: _ClassVar[int]
    POLICY_TIMESTEP_US_FIELD_NUMBER: _ClassVar[int]
    REAR_AXLE_OFFSET_M_FIELD_NUMBER: _ClassVar[int]
    CAMERAS_FIELD_NUMBER: _ClassVar[int]
    base: _egodriver_pb2.DriveSessionRequest
    map_name: str
    scenario_name: str
    fixed_delta_us: int
    policy_timestep_us: int
    rear_axle_offset_m: float
    cameras: _containers.RepeatedCompositeFieldContainer[_sensorsim_pb2.AvailableCamerasReturn.AvailableCamera]
    def __init__(self, base: _Optional[_Union[_egodriver_pb2.DriveSessionRequest, _Mapping]] = ..., map_name: _Optional[str] = ..., scenario_name: _Optional[str] = ..., fixed_delta_us: _Optional[int] = ..., policy_timestep_us: _Optional[int] = ..., rear_axle_offset_m: _Optional[float] = ..., cameras: _Optional[_Iterable[_Union[_sensorsim_pb2.AvailableCamerasReturn.AvailableCamera, _Mapping]]] = ...) -> None: ...

class CompatEntry(_message.Message):
    __slots__ = ("area", "alpasim_behaviour", "implementation_behaviour", "level", "symbols")
    AREA_FIELD_NUMBER: _ClassVar[int]
    ALPASIM_BEHAVIOUR_FIELD_NUMBER: _ClassVar[int]
    IMPLEMENTATION_BEHAVIOUR_FIELD_NUMBER: _ClassVar[int]
    LEVEL_FIELD_NUMBER: _ClassVar[int]
    SYMBOLS_FIELD_NUMBER: _ClassVar[int]
    area: str
    alpasim_behaviour: str
    implementation_behaviour: str
    level: CompatLevel
    symbols: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, area: _Optional[str] = ..., alpasim_behaviour: _Optional[str] = ..., implementation_behaviour: _Optional[str] = ..., level: _Optional[_Union[CompatLevel, str]] = ..., symbols: _Optional[_Iterable[str]] = ...) -> None: ...

class CompatReport(_message.Message):
    __slots__ = ("implementation_version", "alpasim_grpc_rev", "alpasim_grpc_api_version", "entries")
    IMPLEMENTATION_VERSION_FIELD_NUMBER: _ClassVar[int]
    ALPASIM_GRPC_REV_FIELD_NUMBER: _ClassVar[int]
    ALPASIM_GRPC_API_VERSION_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    implementation_version: str
    alpasim_grpc_rev: str
    alpasim_grpc_api_version: _common_pb2.VersionId.APIVersion
    entries: _containers.RepeatedCompositeFieldContainer[CompatEntry]
    def __init__(self, implementation_version: _Optional[str] = ..., alpasim_grpc_rev: _Optional[str] = ..., alpasim_grpc_api_version: _Optional[_Union[_common_pb2.VersionId.APIVersion, _Mapping]] = ..., entries: _Optional[_Iterable[_Union[CompatEntry, _Mapping]]] = ...) -> None: ...
