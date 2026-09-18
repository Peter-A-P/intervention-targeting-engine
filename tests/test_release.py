"""The things a release claims about itself have to agree with each other.

Three numbers and one file say what this is: the version in `pyproject.toml`, the version
`itx --version` prints, the licence the package metadata declares, and the licence text in
the repository. Each is edited by hand in a different place, which is how they drift, and
the drift is invisible until somebody installs the wheel and finds it says one thing while
the repository says another.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import itx

PYPROJECT = Path("pyproject.toml")
LICENCE = Path("LICENSE")


def _metadata() -> dict[str, object]:
    with PYPROJECT.open("rb") as handle:
        project: dict[str, object] = tomllib.load(handle)["project"]
    return project


def test_the_package_and_the_project_agree_on_the_version() -> None:
    assert itx.__version__ == _metadata()["version"]


def test_the_version_is_a_release_rather_than_a_development_placeholder() -> None:
    # 0.1.0.dev0 was right for eight weeks of building and is wrong on a tagged release:
    # anyone installing it would get a version that sorts below the tag it came from.
    version = str(_metadata()["version"])
    assert not version.endswith(("dev0", ".dev")), version
    assert "dev" not in version, version


def test_the_declared_licence_is_in_the_repository() -> None:
    # A licence expression in metadata with no text beside it is a claim with nothing
    # behind it: GitHub shows no licence, and a reader has nothing to read.
    assert _metadata()["license"] == "MIT"
    assert _metadata()["license-files"] == ["LICENSE"]
    text = LICENCE.read_text(encoding="utf-8")
    assert text.startswith("MIT License")
    assert "Peter Parker" in text


def test_the_readme_points_at_the_licence_it_ships() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "## Licence" in readme
    assert "[LICENSE](LICENSE)" in readme
