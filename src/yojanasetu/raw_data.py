"""Validation of scheme folders under data/raw/.

Each folder holds a meta.yaml (provenance) and a source.txt (official text,
pasted verbatim). This module checks both and records a hash of the text so we
can later detect that a source changed after rules were written from it.

Design note: raw text is NEVER modified here. Unicode normalization (NFC) is
applied later, at ingestion time, so data/raw always stays exactly as pasted.
"""

import codecs
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from yojanasetu.schemas.source_meta import SchemeSourceMeta, load_meta

PLACEHOLDER_MARKER = "PASTE THE OFFICIAL TEXT HERE"


@dataclass
class FolderReport:
    """Result of validating one scheme folder."""

    folder: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    meta: SchemeSourceMeta | None = None
    sha256: str = ""

    @property
    def ok(self) -> bool:
        """True when there are no errors (warnings are allowed)."""
        return not self.errors


def compute_sha256(text: str) -> str:
    """Hash text with Windows/Unix line endings treated as identical.

    Why: git may convert line endings between machines. Without this, the same
    text would produce different hashes and trigger false "source changed" alarms.
    """
    canonical = text.replace("\r\n", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def find_scheme_folders(raw_dir: Path) -> list[Path]:
    """List scheme folders, skipping _TEMPLATE and hidden folders."""
    return sorted(
        p for p in raw_dir.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))
    )


def validate_scheme_folder(folder: Path) -> FolderReport:
    """Check one scheme folder and return every problem found."""
    report = FolderReport(folder=folder)
    meta_path = folder / "meta.yaml"
    source_path = folder / "source.txt"

    if not meta_path.is_file():
        report.errors.append("meta.yaml is missing")
    else:
        try:
            report.meta = load_meta(meta_path)
        except (ValueError, yaml.YAMLError) as exc:
            report.errors.append(f"meta.yaml is invalid: {exc}")

    if report.meta is not None and report.meta.scheme_id != folder.name:
        report.errors.append(
            f"scheme_id '{report.meta.scheme_id}' does not match folder name '{folder.name}'"
        )

    if not source_path.is_file():
        report.errors.append("source.txt is missing")
        return report

    raw_bytes = source_path.read_bytes()
    if raw_bytes.startswith(codecs.BOM_UTF8):
        report.warnings.append("source.txt starts with a UTF-8 BOM (harmless, but re-save as plain UTF-8)")
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        report.errors.append("source.txt is not valid UTF-8 (re-save the file as UTF-8)")
        return report

    if not text.strip():
        report.errors.append("source.txt is empty")
        return report
    if PLACEHOLDER_MARKER in text:
        report.errors.append("source.txt still contains the template placeholder text")
    if "\ufffd" in text:
        report.errors.append("source.txt contains broken characters (U+FFFD); the paste was corrupted")
    if not unicodedata.is_normalized("NFC", text):
        report.warnings.append(
            "source.txt is not Unicode NFC. Left as pasted; it will be normalized at ingestion."
        )

    report.sha256 = compute_sha256(text)
    if report.meta is not None:
        if not report.meta.source_sha256:
            report.warnings.append("no source_sha256 recorded yet (run with --write-hash)")
        elif report.meta.source_sha256 != report.sha256:
            report.errors.append(
                "source.txt changed since its hash was recorded. If the change is "
                "deliberate, clear source_sha256 in meta.yaml and re-run with --write-hash; "
                "then re-check every rule written from this scheme."
            )
    return report


def write_hash(meta_path: Path, sha256: str) -> None:
    """Write the hash into meta.yaml, keeping the rest of the file (and comments) intact."""
    content = meta_path.read_text(encoding="utf-8")
    line = f'source_sha256: "{sha256}"  '
    if re.search(r"^source_sha256:", content, flags=re.MULTILINE):
        content = re.sub(
            r"^source_sha256:[^#\n]*", lambda _match: line, content, flags=re.MULTILINE
        )
    else:
        content = content.rstrip("\n") + "\n" + line.rstrip() + "\n"
    meta_path.write_text(content, encoding="utf-8")
