"""Loaders: download, verify, encode, split. Nothing raw is ever committed."""

from itx.data.acic import load_acic, load_acic_replicates
from itx.data.download import ChecksumMismatchError, fetch, fetch_all, sha256_of
from itx.data.hillstrom import load_hillstrom
from itx.data.ihdp import load_ihdp, load_ihdp_replicates
from itx.data.registry import SOURCES, Source, checksums, data_dir, source
from itx.data.splits import stratified_split

__all__ = [
    "SOURCES",
    "ChecksumMismatchError",
    "Source",
    "checksums",
    "data_dir",
    "fetch",
    "fetch_all",
    "load_acic",
    "load_acic_replicates",
    "load_hillstrom",
    "load_ihdp",
    "load_ihdp_replicates",
    "sha256_of",
    "source",
    "stratified_split",
]
