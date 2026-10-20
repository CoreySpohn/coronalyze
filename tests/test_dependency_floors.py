"""The packaging metadata must keep old and new library releases apart.

The sky convention changed across orbix 3, skyscapes 2.1 and coronagraphoto
4.2, so an install that mixes an older release with this one would silently
place sources on the wrong side of the star. These tests read pyproject.toml
and assert the minimum versions that every path to coronagraphoto declares.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"
CORONAGRAPHOTO_FLOOR = Version("4.2.0")


def _lower_bound(requirement):
    """Return the smallest version a requirement admits, or None if unbounded."""
    bounds = [
        Version(spec.version)
        for spec in requirement.specifier
        if spec.operator in (">=", "==", "~=", ">")
    ]
    return max(bounds) if bounds else None


def _coronagraphoto_requirements():
    """Yield (where, requirement) for every coronagraphoto pin in the metadata."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    groups = {"dependencies": project.get("dependencies", [])}
    for name, reqs in project.get("optional-dependencies", {}).items():
        groups[f"extra {name}"] = reqs
    for where, reqs in groups.items():
        for text in reqs:
            req = Requirement(text)
            if req.name == "coronagraphoto":
                yield where, req


def test_every_coronagraphoto_pin_has_the_floor():
    """Each extra that pulls in coronagraphoto requires the convention release."""
    found = dict(_coronagraphoto_requirements())
    assert {"extra coronagraphoto", "extra docs"} <= set(found)
    for where, req in found.items():
        floor = _lower_bound(req)
        assert floor is not None and floor >= CORONAGRAPHOTO_FLOOR, (
            f"{where}: {req} admits a coronagraphoto older than {CORONAGRAPHOTO_FLOOR}"
        )


def test_python_floor_matches_coronagraphoto():
    """Coronalyze must not admit an interpreter coronagraphoto cannot run on."""
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    spec = SpecifierSet(project["requires-python"])
    assert "3.10" not in spec, f"requires-python {spec} admits Python 3.10"
    assert "3.11" in spec
