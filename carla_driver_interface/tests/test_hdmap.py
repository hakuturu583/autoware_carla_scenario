# SPDX-License-Identifier: Apache-2.0
"""The map as files: written once by the runtime, resolved by the driver half.

The map is a roadgen-built road with one traffic light, written out as
OpenDRIVE: the same path a CARLA world's ``to_opendrive()`` takes, without a
CARLA server. What is pinned is the chain a light travels -- OpenDRIVE signal
id and lane, to the Lanelet2 lanelet and regulatory element a driver reads.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from dataclasses import replace
from pathlib import Path

import pytest

roadgen = pytest.importorskip("roadgen")

from carla_driver_interface.driver.base import BaseDriver, DriveContext, DriveResult  # noqa: E402
from carla_driver_interface.driver.policies import RouteFollowerPolicy  # noqa: E402
from carla_driver_interface.driver.server import serving  # noqa: E402
from carla_driver_interface.fakes import FakeWorld  # noqa: E402
from carla_driver_interface.grpc_api import (  # noqa: E402
    ImageFormat,
    StopPoint,
    TrafficLight,
    TrafficLightState,
    Vec3,
)
from carla_driver_interface.hdmap import MapFiles  # noqa: E402
from carla_driver_interface.hdmap.export import (  # noqa: E402
    MANIFEST_FILE,
    export_map,
    map_id_for,
    sanitize_opendrive,
)
from carla_driver_interface.runtime import CarlaRuntime, RuntimeConfig, ScenarioSpec  # noqa: E402


@pytest.fixture(scope="module")
def opendrive(tmp_path_factory) -> str:
    """Two roads through a junction; a light where the first one ends."""
    m = roadgen.Map()

    def lanes():
        return [
            roadgen.Lane(width=3.5, direction="forward"),
            roadgen.Lane(width=3.5, direction="backward"),
        ]

    a = m.add_road(start=(0.0, 0.0, 0.0), end=(100.0, 0.0, 0.0), lanes=lanes())
    b = m.add_road(start=(120.0, 0.0, 0.0), end=(220.0, 0.0, 0.0), lanes=lanes())
    m.connect(a, b, junction=m.add_junction())
    light = m.add_traffic_light(a.lane(0), end="end")
    stop = m.add_stop_line(a.lane(0), end="end")
    m.add_traffic_light_rule([light], [a.lane(0)], stop_line=stop)
    path = tmp_path_factory.mktemp("xodr") / "town.xodr"
    m.export_opendrive(str(path))
    return path.read_text(encoding="utf-8")


def _signal(opendrive: str) -> ElementTree.Element:
    return next(ElementTree.fromstring(opendrive).iter("signal"))


def _light(opendrive: str, state=TrafficLightState.TRAFFIC_LIGHT_STATE_RED) -> TrafficLight:
    """The light as the runtime reports it: its id, a stop point on the first
    road's right-hand lane (OpenDRIVE lane -1), at the road's end."""
    return TrafficLight(
        opendrive_id=_signal(opendrive).get("id"),
        state=state,
        stop_points=[
            StopPoint(
                road_id=0,
                section_id=0,
                lane_id=-1,
                position_local=Vec3(x=99.0, y=-1.75, z=0.0),
            )
        ],
    )


def _relations(osm: Path) -> dict[str, ElementTree.Element]:
    return {rel.get("id"): rel for rel in ElementTree.parse(osm).getroot().iter("relation")}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_the_map_is_written_once_under_its_content_id(opendrive, tmp_path):
    map_id = export_map(opendrive, "Carla/Maps/Town", tmp_path)
    assert map_id == map_id_for("Carla/Maps/Town", opendrive)
    assert map_id.startswith("Town-")
    directory = tmp_path / map_id
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    assert set(manifest["formats"]) == {"lanelet2"}
    assert (directory / manifest["formats"]["lanelet2"]["path"]).is_file()
    assert (directory / manifest["source"]).read_text() == opendrive

    written = (directory / MANIFEST_FILE).stat().st_mtime_ns
    assert export_map(opendrive, "Carla/Maps/Town", tmp_path) == map_id
    assert (directory / MANIFEST_FILE).stat().st_mtime_ns == written, "rewrote a complete set"
    assert [p.name for p in tmp_path.iterdir()] == [map_id], "left scratch behind"


def test_a_changed_map_gets_a_new_id():
    assert map_id_for("Town", "<OpenDRIVE/>") != map_id_for("Town", "<OpenDRIVE />")


def test_an_unknown_format_is_refused_before_anything_is_written(opendrive, tmp_path):
    with pytest.raises(ValueError, match="unknown map format"):
        export_map(opendrive, "Town", tmp_path, formats=("lanelet3",))
    assert not any(tmp_path.iterdir())


def test_carla_departures_from_the_standard_are_rewritten():
    carla = (
        '<header><userData><vectorScene program="RoadRunner"/></userData></header>'
        '<roadMark sOffset="0" type="curb" weight="standard"/>'
        '<object id="1" type="-1" name="x"/>'
        '<cornerLocal u="0" v="0" z="0"/>'
    )
    fixed = sanitize_opendrive(carla)
    assert "userData" not in fixed
    assert 'type="curb" weight="standard" color="standard"' in fixed
    assert 'type="none"' in fixed
    assert 'height="0"' in fixed


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_a_light_resolves_to_its_lanelet_and_regulatory_element(opendrive, tmp_path):
    map_files = MapFiles.open(tmp_path, export_map(opendrive, "Town", tmp_path))
    [stop] = map_files.stop_lines([_light(opendrive)])

    assert stop.traffic_light_id == _signal(opendrive).get("id")
    assert stop.state == TrafficLightState.TRAFFIC_LIGHT_STATE_RED
    assert stop.position_local.tolist() == [99.0, -1.75, 0.0]
    assert len(stop.lane_ids) == 1 and stop.rule_ids and stop.light_ids

    # Cross-check against the Lanelet2 file itself: the lanelet is the one
    # that carries the regulatory element, and the element refers to the light.
    relations = _relations(map_files.path)
    lanelet = relations[stop.lane_ids[0]]
    regulatory = [
        m.get("ref") for m in lanelet.iter("member") if m.get("role") == "regulatory_element"
    ]
    assert set(stop.rule_ids) <= set(regulatory)
    refers = {
        m.get("ref")
        for m in relations[stop.rule_ids[0]].iter("member")
        if m.get("role") == "refers"
    }
    assert refers & set(stop.light_ids)


def test_a_light_the_map_does_not_have_still_reports_where_to_stop(opendrive, tmp_path):
    map_files = MapFiles.open(tmp_path, export_map(opendrive, "Town", tmp_path))
    stray = _light(opendrive)
    stray.opendrive_id = "no-such-signal"
    stray.stop_points[0].lane_id = -7
    [stop] = map_files.stop_lines([stray])
    assert stop.lane_ids == () and stop.rule_ids == () and stop.light_ids == ()
    assert stop.position_local.tolist() == [99.0, -1.75, 0.0]


def test_opening_a_missing_set_or_format_says_what_to_fix(opendrive, tmp_path):
    with pytest.raises(FileNotFoundError, match="map_dir"):
        MapFiles.open(tmp_path, "Town-000000000000")
    map_id = export_map(opendrive, "Town", tmp_path)
    with pytest.raises(ValueError, match="map_formats"):
        MapFiles.open(tmp_path, map_id, format="clipgt")


# ---------------------------------------------------------------------------
# End to end, over gRPC
# ---------------------------------------------------------------------------


class _Reader(BaseDriver):
    """Follows the route, and keeps the map and stop lines it was handed."""

    name = "reader"

    def __init__(self, map_dir: Path) -> None:
        self.map_dir = str(map_dir)
        self._follower = RouteFollowerPolicy()
        self.maps: list[MapFiles | None] = []
        self.stop_lines: list[list] = []

    def drive(self, ctx: DriveContext) -> DriveResult:
        self.maps.append(ctx.map)
        self.stop_lines.append(ctx.stop_lines())
        return self._follower.drive(ctx)


def test_the_driver_reads_the_map_the_runtime_wrote_and_gets_stop_lines(opendrive, tmp_path):
    policy = _Reader(tmp_path)
    scenario = ScenarioSpec(map_name="Town", name="straight")
    with serving(policy, port=0, host="127.0.0.1") as port:
        base = RuntimeConfig(
            driver_address=f"127.0.0.1:{port}",
            max_steps=5,
            image_format=ImageFormat.JPEG,
            map_dir=str(tmp_path),
        )
        cfg = replace(base, cameras=[replace(base.cameras[0], width=32, height=24)])
        world = FakeWorld(cfg, scenario, opendrive=opendrive, traffic_lights=[_light(opendrive)])
        outcome = CarlaRuntime(world, cfg, scenario).run_rollout()

    assert outcome.rollout_return.error == ""
    assert len(policy.maps) == 5
    first = policy.maps[0]
    assert first is not None and first.map_id == map_id_for("Town", opendrive)
    assert all(m is first for m in policy.maps), "the set is opened once, not per step"
    assert all(len(stops) == 1 and stops[0].lane_ids for stops in policy.stop_lines)


def test_a_map_dir_without_an_opendrive_is_refused_at_setup(tmp_path):
    cfg = RuntimeConfig(max_steps=2, map_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="OpenDRIVE"):
        CarlaRuntime(FakeWorld(cfg), cfg).run_rollout()


def test_without_a_map_dir_nothing_about_the_map_is_sent():
    world = FakeWorld(RuntimeConfig(), traffic_lights=[TrafficLight(opendrive_id="1")])
    world.setup()
    data = world.environment(world.tick())
    assert data.map_id == "" and len(data.traffic_lights) == 0
