"""validate_raw_data.py – Phase 0 data-quality gate.

Walks every scheme folder under data/raw/ (skipping _TEMPLATE) and checks:
  1. meta.yaml is present and parses as a valid SchemeSourceMeta.
  2. The scheme_id field equals the folder name.
  3. source.txt is present, non-empty, valid UTF-8, and NFC-normalised.
  4. source.txt does not contain the template placeholder text.
  5. Fills in source_sha256 in meta.yaml (SHA-256 of source.txt bytes).

Exit codes:
  0 – all schemes pass.
  1 – one or more schemes have errors (details printed to stderr).

Usage:
  python scripts/validate_raw_data.py          # check all schemes
  python scripts/validate_raw_data.py --fix    # also write source_sha256 back to meta.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import unicodedata
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TEMPLATE_DIR = RAW_DIR / "_TEMPLATE"

# The placeholder text that must NOT appear in a real source.txt.
TEMPLATE_SOURCE_SENTINEL = "PASTE THE OFFICIAL TEXT HERE, VERBATIM."


def sha256_of_file(path: Path) -> str:
    """Return the lower-hex SHA-256 digest of a file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_source_txt(folder: Path) -> list[str]:
    """Return a list of error strings for source.txt in *folder* (empty = OK)."""
    errors: list[str] = []
    source_path = folder / "source.txt"

    if not source_path.exists():
        return [f"{folder.name}: source.txt is missing"]

    # Must be valid UTF-8.
    try:
        raw_bytes = source_path.read_bytes()
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"{folder.name}: source.txt is not valid UTF-8: {exc}"]

    # Must be non-empty (after stripping whitespace).
    if not text.strip():
        errors.append(f"{folder.name}: source.txt is empty or whitespace-only")

    # Must not be the template placeholder.
    if TEMPLATE_SOURCE_SENTINEL in text:
        errors.append(
            f"{folder.name}: source.txt still contains the template placeholder text"
        )

    # Must be NFC-normalised (AGENTS.md rule: normalise Hindi text to NFC on ingest).
    normalised = unicodedata.normalize("NFC", text)
    if normalised != text:
        errors.append(
            f"{folder.name}: source.txt is not NFC-normalised "
            "(run unicodedata.normalize('NFC', text) and re-save)"
        )

    return errors


def check_meta_yaml(folder: Path) -> tuple[list[str], str]:
    """Validate meta.yaml; return (errors, sha256_of_source_txt).

    We import SchemeSourceMeta here (inside the function) so that import errors
    surface as a clear message rather than a module-level crash.
    """
    errors: list[str] = []
    source_sha256 = ""

    meta_path = folder / "meta.yaml"
    if not meta_path.exists():
        return [f"{folder.name}: meta.yaml is missing"], source_sha256

    # Load raw YAML first so we can check scheme_id before Pydantic runs.
    try:
        raw = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return [f"{folder.name}: meta.yaml is not valid YAML: {exc}"], source_sha256

    if not isinstance(raw, dict):
        return [f"{folder.name}: meta.yaml must be a YAML mapping (key: value)"], source_sha256

    # scheme_id must equal the folder name.
    if raw.get("scheme_id") != folder.name:
        errors.append(
            f"{folder.name}: meta.yaml scheme_id '{raw.get('scheme_id')}' "
            f"does not match the folder name '{folder.name}'"
        )

    # Compute SHA-256 of source.txt (if it exists) so we can compare / fill it.
    source_path = folder / "source.txt"
    if source_path.exists():
        source_sha256 = sha256_of_file(source_path)
        stored = raw.get("source_sha256", "")
        if stored and stored != source_sha256:
            errors.append(
                f"{folder.name}: source_sha256 in meta.yaml does not match "
                f"the actual SHA-256 of source.txt "
                f"(stored={stored[:12]}…, actual={source_sha256[:12]}…)"
            )

    # Full Pydantic validation via SchemeSourceMeta.
    try:
        from yojanasetu.schemas.source_meta import SchemeSourceMeta  # noqa: PLC0415

        SchemeSourceMeta.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError or ValueError
        errors.append(f"{folder.name}: meta.yaml failed validation: {exc}")

    return errors, source_sha256


def fill_sha256(meta_path: Path, sha256: str) -> None:
    """Write source_sha256 back into meta.yaml, preserving all other content.

    We do a targeted in-place replacement rather than a full YAML round-trip so
    that comments and formatting in the file are kept intact.
    """
    text = meta_path.read_text(encoding="utf-8")
    old_line = 'source_sha256: ""'
    new_line = f'source_sha256: "{sha256}"'
    if old_line in text:
        text = text.replace(old_line, new_line, 1)
        meta_path.write_text(text, encoding="utf-8")
    elif 'source_sha256: "' not in text:
        # Field is absent – append it.
        text = text.rstrip() + f"\nsource_sha256: \"{sha256}\"\n"
        meta_path.write_text(text, encoding="utf-8")
    # If it is already non-empty and correct, do nothing.


def validate_all(fix: bool = False) -> int:
    """Run all checks. Return 0 on success, 1 if any error was found."""
    if not RAW_DIR.exists():
        print(f"ERROR: data/raw/ directory not found at {RAW_DIR}", file=sys.stderr)
        return 1

    scheme_folders = sorted(
        d for d in RAW_DIR.iterdir()
        if d.is_dir() and d.name != "_TEMPLATE"
    )

    if not scheme_folders:
        print(
            "WARNING: No scheme folders found under data/raw/ "
            "(only _TEMPLATE exists). Nothing to validate.",
            file=sys.stderr,
        )
        return 0

    all_errors: list[str] = []
    filled: list[str] = []

    for folder in scheme_folders:
        # 1. Check source.txt.
        src_errors = check_source_txt(folder)
        all_errors.extend(src_errors)

        # 2. Check meta.yaml.
        meta_errors, sha256 = check_meta_yaml(folder)
        all_errors.extend(meta_errors)

        # 3. Fill source_sha256 if requested and no errors so far for this folder.
        folder_errors = src_errors + meta_errors
        if fix and sha256 and not folder_errors:
            meta_path = folder / "meta.yaml"
            raw = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
            if raw.get("source_sha256", "") != sha256:
                fill_sha256(meta_path, sha256)
                filled.append(folder.name)

    if filled:
        print(f"Filled source_sha256 for: {', '.join(filled)}")

    if all_errors:
        print("\nValidation FAILED. Errors found:", file=sys.stderr)
        for err in all_errors:
            print(f"  ✗ {err}", file=sys.stderr)
        return 1

    print(f"All {len(scheme_folders)} scheme(s) passed validation. ✓")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Write source_sha256 back into meta.yaml for passing schemes.",
    )
    args = parser.parse_args()
    sys.exit(validate_all(fix=args.fix))


if __name__ == "__main__":
    main()
