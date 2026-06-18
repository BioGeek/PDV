#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


SEQUENCE_COLUMNS = [
    "diffusion_predictions_tokenised",
    "predictions_tokenised",
    "preds_tokenised",
    "transformer_predictions_tokenised",
    "diffusion_predictions",
    "predictions",
    "preds",
    "transformer_predictions",
]

SCORE_COLUMNS = [
    "log_probs",
    "log_probabilities",
    "prediction_log_probability",
    "diffusion_log_probabilities",
    "transformer_log_probabilities",
    "delta_mass_ppm",
]


@dataclass
class ParsedPeptide:
    sequence: str
    modification_count: int


@dataclass
class ValidationResult:
    path: Path
    rows: int
    accepted_rows: int
    sequence_columns: set[str]
    score_columns: set[str]
    max_modifications: int
    error: str | None = None


def clean_token(token: str) -> str:
    token = token.strip()
    if (token.startswith("'") and token.endswith("'")) or (token.startswith('"') and token.endswith('"')):
        token = token[1:-1]
    return token.strip()


def get_modification_mass(token: str) -> float | None:
    token = token.strip()
    if token.startswith("[") and token.endswith("]"):
        token = token[1:-1]
    if token.startswith("(") and token.endswith(")"):
        token = token[1:-1]

    unimod = {
        "UNIMOD:1": 42.010565,
        "UNIMOD:4": 57.021464,
        "UNIMOD:5": 43.005814,
        "UNIMOD:7": 0.984016,
        "UNIMOD:21": 79.966331,
        "UNIMOD:35": 15.994915,
        "UNIMOD:385": -17.026549,
    }
    upper = token.upper()
    if upper in unimod:
        return unimod[upper]
    if upper == "OX":
        return 15.994915
    if upper == "P":
        return 79.966331

    try:
        return float(token)
    except ValueError:
        return None


def parse_tokens(tokens: list[str]) -> ParsedPeptide:
    sequence: list[str] = []
    modification_count = 0
    pending_n_term_mods = 0

    for token in tokens:
        if not token or token.lower() == "nan":
            continue
        if token[0].isalpha():
            sequence.append(token[0].upper())
            if len(token) > 1 and get_modification_mass(token[1:]) is not None:
                modification_count += 1
        elif get_modification_mass(token) is not None:
            pending_n_term_mods += 1

    return ParsedPeptide("".join(sequence), modification_count + pending_n_term_mods)


def parse_tokenised_peptide(value: str) -> ParsedPeptide:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]

    if "," not in value:
        return parse_peptide_string(value)

    return parse_tokens([clean_token(token) for token in value.split(",")])


def parse_peptide_string(value: str) -> ParsedPeptide:
    value = value.strip()
    if value.startswith("_") and value.endswith("_") and len(value) > 1:
        value = value[1:-1]
    if value.startswith(".") and value.endswith(".") and len(value) > 1:
        value = value[1:-1]

    sequence: list[str] = []
    modification_count = 0
    pending_n_term_mods = 0
    index = 0

    if "-" in value and value and not value[0].isalpha():
        terminal_token = value[: value.index("-")]
        if get_modification_mass(terminal_token) is not None:
            pending_n_term_mods += 1
        index = value.index("-") + 1

    while index < len(value):
        current = value[index]
        if current.isalpha():
            sequence.append(current.upper())
            index += 1
            if index < len(value) and value[index] in "[(":
                close = "]" if value[index] == "[" else ")"
                close_index = value.find(close, index)
                if close_index > index:
                    if get_modification_mass(value[index : close_index + 1]) is not None:
                        modification_count += 1
                    index = close_index + 1
        elif current == "," or current.isspace():
            index += 1
        else:
            index += 1

    return ParsedPeptide("".join(sequence), modification_count + pending_n_term_mods)


def parse_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        return 0.0
    if math.isnan(parsed) or math.isinf(parsed):
        return 0.0
    return parsed


def validate_file(path: Path) -> ValidationResult:
    rows = 0
    accepted = 0
    sequence_columns: set[str] = set()
    score_columns: set[str] = set()
    max_modifications = 0

    try:
        with path.open(newline="") as handle:
            reader = csv.reader(handle)
            try:
                headers = [header.strip().lower().lstrip("\ufeff") for header in next(reader)]
            except StopIteration:
                return ValidationResult(path, 0, 0, set(), set(), 0, "empty file")

            header_map = {header: index for index, header in enumerate(headers)}

            for row in reader:
                rows += 1
                parsed_peptide: ParsedPeptide | None = None

                for column in SEQUENCE_COLUMNS:
                    index = header_map.get(column)
                    if index is None or index >= len(row):
                        continue
                    value = row[index].strip()
                    if not value or value.lower() == "nan":
                        continue
                    parsed_peptide = (
                        parse_tokenised_peptide(value)
                        if column.endswith("tokenised")
                        else parse_peptide_string(value)
                    )
                    if parsed_peptide.sequence:
                        sequence_columns.add(column)
                    break

                for column in SCORE_COLUMNS:
                    index = header_map.get(column)
                    if index is not None and index < len(row) and row[index].strip():
                        parse_float(row[index].strip())
                        score_columns.add(column)
                        break

                if parsed_peptide is not None and parsed_peptide.sequence:
                    accepted += 1
                    max_modifications = max(max_modifications, parsed_peptide.modification_count)
    except Exception as exc:  # noqa: BLE001 - validation should report every file.
        return ValidationResult(path, rows, accepted, sequence_columns, score_columns, max_modifications, str(exc))

    error = None if accepted > 0 else "no loadable InstaNovo predictions"
    return ValidationResult(path, rows, accepted, sequence_columns, score_columns, max_modifications, error)


def iter_prediction_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.csv")))
        elif path.is_file() and path.suffix.lower() == ".csv":
            files.append(path)

    return [
        path
        for path in files
        if path.name != "prediction_manifest.csv" and not path.name.startswith(".")
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate InstaNovo CSVs against the PDV InstaNovoImport parsing rules."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    files = iter_prediction_files(args.paths)
    if not files:
        print("No prediction CSV files found.")
        return 1

    failures = 0
    for result in [validate_file(path) for path in files]:
        status = "OK" if result.error is None else "FAIL"
        if result.error is not None:
            failures += 1
        print(
            "\t".join(
                [
                    status,
                    str(result.path),
                    f"rows={result.rows}",
                    f"accepted={result.accepted_rows}",
                    "sequence_columns=" + ",".join(sorted(result.sequence_columns)),
                    "score_columns=" + ",".join(sorted(result.score_columns)),
                    f"max_modifications={result.max_modifications}",
                    f"error={result.error or ''}",
                ]
            )
        )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
