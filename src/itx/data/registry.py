"""Where each raw file comes from, and the digest it has to match.

Nothing raw is committed (PLAN.md section 3). The checksums are, in
``src/itx/data/checksums.sha256``, in the format ``sha256sum -c`` reads, so a stranger
can verify a download without running any of this code::

    cd data/raw && sha256sum -c ../../src/itx/data/checksums.sha256
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path

CHECKSUM_RESOURCE = "checksums.sha256"


@dataclass(frozen=True, slots=True)
class Source:
    """One downloadable raw file.

    Attributes:
        key: Identifier used by ``itx data pull``.
        filename: Name the file is cached under, and the name in the checksum file.
        url: Where it is fetched from.
        licence: The terms the file is published under, repeated in its dataset card.
        note: Anything a reader needs to know before downloading, such as a click-through.
    """

    key: str
    filename: str
    url: str
    licence: str
    note: str = ""

    @property
    def expected_sha256(self) -> str:
        """The committed digest for this file."""
        digests = checksums()
        if self.filename not in digests:
            msg = (
                f"no committed checksum for {self.filename}; "
                f"add one with 'itx data checksum {self.key}' after verifying the download"
            )
            raise KeyError(msg)
        return digests[self.filename]


#: The causallib mirror of the ACIC 2016 competition data. The competition's own
#: distribution is an R package; this mirror is plain CSV, is maintained, and carries the
#: organisers' licence and citation alongside the files.
ACIC_BASE = (
    "https://raw.githubusercontent.com/BiomedSciAI/causallib/master/"
    "causallib/datasets/data/acic_challenge_2016"
)
ACIC_REPLICATES = 10
ACIC_LICENCE = "Community Data License Agreement - Sharing, Version 1.0"


def _acic_sources() -> dict[str, Source]:
    """The shared covariate file plus one file per simulation setting."""
    sources = {
        "acic-x": Source(
            key="acic-x",
            filename="acic2016_x.csv",
            url=f"{ACIC_BASE}/x.csv",
            licence=ACIC_LICENCE,
            note="4,802 units, 58 real covariates, shared by every replicate.",
        )
    }
    for index in range(1, ACIC_REPLICATES + 1):
        sources[f"acic-zymu-{index}"] = Source(
            key=f"acic-zymu-{index}",
            filename=f"acic2016_zymu_{index}.csv",
            url=f"{ACIC_BASE}/zymu_{index}.csv",
            licence=ACIC_LICENCE,
            note=f"Simulation setting {index}: treatment, both potential outcomes, mu0, mu1.",
        )
    return sources


SOURCES: dict[str, Source] = {
    "hillstrom": Source(
        key="hillstrom",
        filename="hillstrom.csv",
        url=(
            "http://www.minethatdata.com/"
            "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
        ),
        licence=(
            "No formal licence stated; published for public use, attributed to MineThatData"
        ),
        note="64,000 customers, a three-arm email experiment. See docs/data/hillstrom.md.",
    ),
    "criteo": Source(
        key="criteo",
        filename="criteo-research-uplift-v2.1.csv.gz",
        url="http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
        licence="CC BY-NC-SA 4.0 (Criteo research licence)",
        note="297 MB compressed, 13.9M rows. See docs/data/criteo.md.",
    ),
    "lenta": Source(
        key="lenta",
        filename="lenta_dataset.csv.gz",
        url="https://sklift.s3.eu-west-2.amazonaws.com/lenta_dataset.csv.gz",
        licence=(
            "None stated. Not by the publisher, not in scikit-uplift's code or docs. "
            "Downloaded at run time and redistributed nowhere; see docs/data/lenta.md"
        ),
        note="138 MB compressed, 687,029 rows, 194 columns. See docs/data/lenta.md.",
    ),
    "ihdp-train": Source(
        key="ihdp-train",
        filename="ihdp_npci_1-100.train.npz",
        url="https://www.fredjo.com/files/ihdp_npci_1-100.train.npz",
        licence="Public benchmark file, redistributed by the CFR/CEVAE authors",
        note="672 units x 100 replicates. Rejoined with the test file. See docs/data/ihdp.md.",
    ),
    "ihdp-test": Source(
        key="ihdp-test",
        filename="ihdp_npci_1-100.test.npz",
        url="https://www.fredjo.com/files/ihdp_npci_1-100.test.npz",
        licence="Public benchmark file, redistributed by the CFR/CEVAE authors",
        note="75 units x 100 replicates. Rejoined with the train file. See docs/data/ihdp.md.",
    ),
    **_acic_sources(),
}


@cache
def checksums() -> dict[str, str]:
    """Read the committed digests, keyed by filename.

    Returns:
        Mapping from filename to lowercase hex SHA-256.
    """
    text = resources.files("itx.data").joinpath(CHECKSUM_RESOURCE).read_text(encoding="utf-8")
    digests: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, name = line.partition(" ")
        digests[name.strip().lstrip("*")] = digest
    return digests


def source(key: str) -> Source:
    """Look up a source by key.

    Args:
        key: One of the keys in ``SOURCES``.

    Returns:
        The source definition.

    Raises:
        KeyError: If the key is not registered.
    """
    if key not in SOURCES:
        msg = f"unknown data source {key!r}; known: {', '.join(sorted(SOURCES))}"
        raise KeyError(msg)
    return SOURCES[key]


def data_dir() -> Path:
    """Directory raw downloads are cached in.

    ``ITX_DATA_DIR`` wins if set. Inside a source checkout the default is ``data/raw`` at
    the repository root, which is gitignored. Outside one it is a per-user cache
    directory, so an installed copy does not write into the current working directory.

    Returns:
        The cache directory, which may not exist yet.
    """
    override = os.environ.get("ITX_DATA_DIR")
    if override:
        return Path(override)
    root = _repo_root()
    if root is not None:
        return root / "data" / "raw"
    return _user_cache_dir() / "raw"


def _repo_root() -> Path | None:
    """The checkout root, found by walking up from this file for ``pyproject.toml``."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "itx").is_dir():
            return parent
    return None


def _user_cache_dir() -> Path:
    """Per-user cache directory for an installed, non-checkout copy.

    Tested against ``os.name`` rather than ``sys.platform`` so that a type checker running
    on one platform does not decide the other branch is dead code.
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) / "itx" if base else Path.home() / "AppData" / "Local" / "itx"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "itx"
