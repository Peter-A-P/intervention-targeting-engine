"""Download a raw file once, verify its SHA-256, cache it, never commit it."""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from itx.data.registry import Source, data_dir, source

if TYPE_CHECKING:
    from collections.abc import Iterable

CHUNK_BYTES = 1 << 20
USER_AGENT = "itx/0.1 (+https://github.com/Peter-A-P/intervention-targeting-engine)"


class ChecksumMismatchError(RuntimeError):
    """A downloaded or cached file does not match its committed digest."""


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks so a 300 MB download stays out of memory.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(key: str, *, force: bool = False, quiet: bool = False) -> Path:
    """Return a local path to the raw file for ``key``, downloading it if needed.

    A cached file is re-hashed and re-downloaded if it does not match, so a truncated or
    tampered cache cannot silently feed the benchmark.

    Args:
        key: A key from ``registry.SOURCES``.
        force: Re-download even when a matching file is already cached.
        quiet: Suppress progress output.

    Returns:
        Path to the verified file.

    Raises:
        ChecksumMismatchError: If the download does not match the committed digest.
    """
    spec = source(key)
    target = data_dir() / spec.filename
    expected = spec.expected_sha256

    if target.exists() and not force:
        actual = sha256_of(target)
        if actual == expected:
            _say(f"{spec.filename}: cached, checksum ok", quiet=quiet)
            return target
        _say(
            f"{spec.filename}: cached copy has digest {actual[:12]}..., "
            f"expected {expected[:12]}...; re-downloading",
            quiet=quiet,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    _say(f"{spec.filename}: downloading from {spec.url}", quiet=quiet)
    _download(spec.url, partial, quiet=quiet)

    actual = sha256_of(partial)
    if actual != expected:
        partial.unlink(missing_ok=True)
        msg = (
            f"{spec.filename}: downloaded digest {actual} does not match the committed "
            f"{expected}. The source may have changed; do not use this file."
        )
        raise ChecksumMismatchError(msg)

    partial.replace(target)
    _say(f"{spec.filename}: checksum ok, cached at {target}", quiet=quiet)
    return target


def fetch_all(keys: Iterable[str], *, force: bool = False, quiet: bool = False) -> list[Path]:
    """Fetch several sources in order.

    Args:
        keys: Source keys.
        force: Re-download even when cached.
        quiet: Suppress progress output.

    Returns:
        The verified local paths, in the order requested.
    """
    return [fetch(key, force=force, quiet=quiet) for key in keys]


def checksum_line(spec: Source, path: Path) -> str:
    """The ``sha256sum`` line to paste into the committed checksum file.

    Args:
        spec: The source the file came from.
        path: The downloaded file.

    Returns:
        A ``<digest>  <filename>`` line.
    """
    return f"{sha256_of(path)}  {spec.filename}"


def _download(url: str, destination: Path, *, quiet: bool) -> None:
    """Stream ``url`` to ``destination``, printing progress every 16 MB."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request) as response:
        total = int(response.headers.get("Content-Length") or 0)
        with destination.open("wb") as handle:
            done = 0
            next_report = 16 * CHUNK_BYTES
            while chunk := response.read(CHUNK_BYTES):
                handle.write(chunk)
                done += len(chunk)
                if not quiet and done >= next_report:
                    _say(_progress(done, total), quiet=quiet)
                    next_report += 16 * CHUNK_BYTES
    if total and destination.stat().st_size != total:
        destination.unlink(missing_ok=True)
        msg = f"{destination.name}: transfer ended early, {done} of {total} bytes"
        raise ChecksumMismatchError(msg)


def _progress(done: int, total: int) -> str:
    """Human-readable progress line."""
    megabytes = done / (1 << 20)
    if total:
        return f"  {megabytes:,.0f} MB of {total / (1 << 20):,.0f} MB ({done / total:.0%})"
    return f"  {megabytes:,.0f} MB"


def _say(message: str, *, quiet: bool) -> None:
    """Print to stderr unless quiet; stdout stays clean for machine-readable output."""
    if not quiet:
        print(message, file=sys.stderr)


def purge(key: str) -> bool:
    """Delete the cached file for ``key``.

    Args:
        key: A key from ``registry.SOURCES``.

    Returns:
        True if a file was removed.
    """
    target = data_dir() / source(key).filename
    if target.exists():
        target.unlink()
        return True
    return False


def disk_usage() -> int:
    """Total bytes currently cached under the data directory."""
    directory = data_dir()
    if not directory.exists():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def human_bytes(count: int) -> str:
    """Format a byte count for a log line."""
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.1f} {unit}"
        size /= 1024
    msg = "unreachable"
    raise AssertionError(msg)


__all__ = [
    "ChecksumMismatchError",
    "checksum_line",
    "disk_usage",
    "fetch",
    "fetch_all",
    "human_bytes",
    "purge",
    "sha256_of",
]
