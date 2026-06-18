#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import html
import importlib.util
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "aichor" / "generate_instanovo_predictions.py"

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


@dataclass(frozen=True)
class JobMeta:
    input_format: str
    output_file: str
    version: str
    job_id: str
    schema_group: str
    model_family: str
    method: str
    refined: bool
    beam_count: int | None


@dataclass
class Prediction:
    scan: str
    peptide: str
    score: float | None
    score_column: str | None
    source_file: str
    meta: JobMeta


def load_runner_module():
    spec = importlib.util.spec_from_file_location("pdv_instanovo_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def infer_model_family(job_id: str) -> str:
    if "combined" in job_id:
        return "combined"
    if "plus" in job_id:
        return "instanovoplus"
    return "transformer"


def infer_method(job_id: str) -> str:
    if "greedy" in job_id:
        return "greedy"
    if "beams" in job_id:
        return "beam-search"
    if "standalone" in job_id:
        return "plus-standalone"
    if "refined" in job_id:
        return "refined"
    return job_id


def infer_beam_count(job_id: str) -> int | None:
    match = re.search(r"beams(\d+)", job_id)
    if match:
        return int(match.group(1))
    if "greedy" in job_id:
        return 1
    return None


def build_metadata() -> dict[str, JobMeta]:
    runner = load_runner_module()
    metadata: dict[str, JobMeta] = {}
    sources = [
        runner.SourceFile("full", "mgf", Path("/workspace/work/SF_200217_U2OS_TiO2_HCD_OT_rep1.full.mgf")),
        runner.SourceFile("full", "mzML", Path("/workspace/work/SF_200217_U2OS_TiO2_HCD_OT_rep1.full.mzML")),
    ]
    for source in sources:
        for job in runner.build_jobs(source):
            metadata[job.output_path.name] = JobMeta(
                input_format=source.source_format.lower(),
                output_file=job.output_path.name,
                version=job.version,
                job_id=job.job_id,
                schema_group=job.schema_group,
                model_family=infer_model_family(job.job_id),
                method=infer_method(job.job_id),
                refined="refined" in job.job_id,
                beam_count=infer_beam_count(job.job_id),
            )
    return metadata


def clean_token(token: str) -> str:
    token = token.strip()
    if (token.startswith("'") and token.endswith("'")) or (token.startswith('"') and token.endswith('"')):
        token = token[1:-1]
    return token.strip()


def modification_mass(token: str) -> float | None:
    token = token.strip()
    if token.startswith("[") and token.endswith("]"):
        token = token[1:-1]
    if token.startswith("(") and token.endswith(")"):
        token = token[1:-1]
    upper = token.upper()
    masses = {
        "UNIMOD:1": 42.010565,
        "UNIMOD:4": 57.021464,
        "UNIMOD:5": 43.005814,
        "UNIMOD:7": 0.984016,
        "UNIMOD:21": 79.966331,
        "UNIMOD:35": 15.994915,
        "UNIMOD:385": -17.026549,
        "OX": 15.994915,
        "P": 79.966331,
    }
    if upper in masses:
        return masses[upper]
    try:
        return float(token)
    except ValueError:
        return None


def format_mass(mass: float) -> str:
    return f"{mass:+.4f}"


def parse_token_list(value: str) -> list[str]:
    value = value.strip()
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [clean_token(str(item)) for item in parsed]
    except Exception:
        pass
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [clean_token(token) for token in value.split(",")]


def tokens_to_peptide(tokens: list[str]) -> str:
    sequence: list[str] = []
    n_term_mods: list[str] = []
    for token in tokens:
        if not token or token.lower() == "nan":
            continue
        if token[0].isalpha():
            aa = token[0].upper()
            suffix = ""
            if len(token) > 1:
                mass = modification_mass(token[1:])
                if mass is not None:
                    suffix = f"[{format_mass(mass)}]"
            sequence.append(aa + suffix)
        else:
            mass = modification_mass(token)
            if mass is not None:
                n_term_mods.append(f"[{format_mass(mass)}]-")
    return "".join(n_term_mods) + "".join(sequence)


def parse_peptide_string(value: str) -> str:
    value = value.strip()
    if value.startswith("_") and value.endswith("_") and len(value) > 1:
        value = value[1:-1]
    if value.startswith(".") and value.endswith(".") and len(value) > 1:
        value = value[1:-1]

    sequence: list[str] = []
    n_term_mods: list[str] = []
    index = 0
    if "-" in value and value and not value[0].isalpha():
        terminal = value[: value.index("-")]
        mass = modification_mass(terminal)
        if mass is not None:
            n_term_mods.append(f"[{format_mass(mass)}]-")
        index = value.index("-") + 1

    while index < len(value):
        current = value[index]
        if current.isalpha():
            aa = current.upper()
            index += 1
            suffix = ""
            if index < len(value) and value[index] in "[(":
                close = "]" if value[index] == "[" else ")"
                close_index = value.find(close, index)
                if close_index > index:
                    mass = modification_mass(value[index : close_index + 1])
                    if mass is not None:
                        suffix = f"[{format_mass(mass)}]"
                    index = close_index + 1
            sequence.append(aa + suffix)
        else:
            index += 1
    return "".join(n_term_mods) + "".join(sequence)


def parse_peptide(value: str, tokenised: bool) -> str:
    if not value or value.lower() == "nan":
        return ""
    return tokens_to_peptide(parse_token_list(value)) if tokenised else parse_peptide_string(value)


def base_sequence(peptide: str) -> str:
    return re.sub(r"\[[^\]]+\]-?", "", peptide)


def peptide_length(peptide: str) -> int:
    return len(base_sequence(peptide))


def peptide_modifications(peptide: str) -> list[str]:
    return re.findall(r"\[([^\]]+)\]", peptide)


def parse_float(value: str) -> float | None:
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def get_scan(row: dict[str, str], fallback: int) -> str:
    for column in ("scan_number", "scan", "index"):
        value = row.get(column, "").strip()
        if value:
            return value.split(".", 1)[0]
    spectrum_id = row.get("spectrum_id", "").strip()
    if spectrum_id:
        match = re.search(r"scan=(\d+)", spectrum_id)
        if match:
            return match.group(1)
        if ":" in spectrum_id:
            return spectrum_id.rsplit(":", 1)[-1].split(".", 1)[0]
        title_match = re.search(r"\.(\d+)\.\d+\.\d+$", spectrum_id)
        if title_match:
            return title_match.group(1)
        return spectrum_id
    return str(fallback)


def read_prediction_file(path: Path, meta: JobMeta) -> tuple[list[Prediction], int]:
    predictions: list[Prediction] = []
    rows = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        reader.fieldnames = [field.strip().lower().lstrip("\ufeff") for field in (reader.fieldnames or [])]
        for row in reader:
            rows += 1
            row = {key.strip().lower(): value for key, value in row.items() if key is not None}
            peptide = ""
            for column in SEQUENCE_COLUMNS:
                value = row.get(column, "").strip()
                if value and value.lower() != "nan":
                    peptide = parse_peptide(value, column.endswith("tokenised"))
                    break
            score = None
            score_column = None
            for column in SCORE_COLUMNS:
                parsed = parse_float(row.get(column, "").strip())
                if parsed is not None:
                    score = parsed
                    score_column = column
                    break
            if peptide:
                predictions.append(
                    Prediction(
                        scan=get_scan(row, rows),
                        peptide=peptide,
                        score=score,
                        score_column=score_column,
                        source_file=path.name,
                        meta=meta,
                    )
                )
    return predictions, rows


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def compare_prediction_maps(
    left: dict[str, Prediction],
    right: dict[str, Prediction],
    max_edit_distance_samples: int,
) -> dict[str, float | int]:
    scans = set(left) | set(right)
    common = sorted(set(left) & set(right))
    left_missing = len(scans - set(left))
    right_missing = len(scans - set(right))
    both_predicted = [scan for scan in common if left[scan].peptide and right[scan].peptide]
    exact = sum(1 for scan in both_predicted if left[scan].peptide == right[scan].peptide)
    mismatch_count = len(both_predicted) - exact
    sample_step = max(1, math.ceil(mismatch_count / max_edit_distance_samples)) if max_edit_distance_samples else 1
    distances = [
        edit_distance(left[scan].peptide, right[scan].peptide)
        for mismatch_index, scan in enumerate(
            scan for scan in both_predicted if left[scan].peptide != right[scan].peptide
        )
        if mismatch_index % sample_step == 0 and mismatch_index // sample_step < max_edit_distance_samples
    ]
    score_diffs = [
        right[scan].score - left[scan].score
        for scan in both_predicted
        if left[scan].score is not None and right[scan].score is not None
    ]
    abs_score_diffs = [abs(value) for value in score_diffs]
    return {
        "total_scans": len(scans),
        "common_predicted": len(both_predicted),
        "exact_matches": exact,
        "exact_match_rate": exact / len(both_predicted) if both_predicted else 0.0,
        "left_missing": left_missing,
        "right_missing": right_missing,
        "left_missing_rate": left_missing / len(scans) if scans else 0.0,
        "right_missing_rate": right_missing / len(scans) if scans else 0.0,
        "mismatch_count": mismatch_count,
        "edit_distance_sample_size": len(distances),
        "mean_mismatch_edit_distance": mean(distances) or 0.0,
        "median_mismatch_edit_distance": quantile(distances, 0.5) or 0.0,
        "median_score_diff": quantile(score_diffs, 0.5) or 0.0,
        "median_abs_score_diff": quantile(abs_score_diffs, 0.5) or 0.0,
        "p90_abs_score_diff": quantile(abs_score_diffs, 0.9) or 0.0,
    }


def even_sample(values: list[object], max_samples: int) -> list[object]:
    if len(values) <= max_samples:
        return values
    step = math.ceil(len(values) / max_samples)
    return [value for index, value in enumerate(values) if index % step == 0][:max_samples]


def comparison_samples(
    left: dict[str, Prediction],
    right: dict[str, Prediction],
    max_samples: int,
) -> tuple[list[int], list[float]]:
    common = sorted(set(left) & set(right))
    mismatches = [scan for scan in common if left[scan].peptide and right[scan].peptide and left[scan].peptide != right[scan].peptide]
    score_scans = [
        scan
        for scan in common
        if left[scan].peptide
        and right[scan].peptide
        and left[scan].score is not None
        and right[scan].score is not None
    ]
    edit_distances = [
        edit_distance(left[scan].peptide, right[scan].peptide)
        for scan in even_sample(mismatches, max_samples)
    ]
    score_diffs = [
        right[scan].score - left[scan].score
        for scan in even_sample(score_scans, max_samples)
    ]
    return edit_distances, score_diffs


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def svg_begin(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:11px;fill:#222}.title{font-size:16px;font-weight:bold}.axis{stroke:#555;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.label{font-size:10px}</style>',
    ]


def svg_text(x: float, y: float, text: object, cls: str = "", anchor: str = "start", rotate: float | None = None) -> str:
    value = html.escape(str(text))
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    class_attr = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{class_attr}{transform}>{value}</text>'


def color(index: int) -> str:
    palette = ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099", "#0099c6", "#dd4477", "#66aa00"]
    return palette[index % len(palette)]


def bar_chart(path: Path, title: str, rows: list[dict[str, object]], value_key: str, label_key: str, group_key: str | None = None, ylabel: str = "") -> None:
    if not rows:
        return
    width = max(1000, 80 + 28 * len(rows))
    height = 560
    left, right, top, bottom = 80, 30, 55, 190
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_value = max(float(row[value_key]) for row in rows) or 1.0
    lines = svg_begin(width, height)
    lines.append(svg_text(width / 2, 25, title, "title", "middle"))
    lines.append(svg_text(16, top + plot_h / 2, ylabel, "label", "middle", -90))
    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    for tick in range(6):
        value = max_value * tick / 5
        y = top + plot_h - (value / max_value) * plot_h
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(svg_text(left - 8, y + 4, f"{value:.2f}", "label", "end"))
    step = plot_w / len(rows)
    bar_w = max(5, step * 0.72)
    groups = {value: idx for idx, value in enumerate(sorted({row.get(group_key, "") for row in rows}))} if group_key else {}
    for idx, row in enumerate(rows):
        value = float(row[value_key])
        x = left + idx * step + (step - bar_w) / 2
        h = (value / max_value) * plot_h if max_value else 0
        y = top + plot_h - h
        fill = color(groups.get(row.get(group_key, ""), idx))
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{fill}"/>')
        lines.append(svg_text(x + bar_w / 2, top + plot_h + 12, row[label_key], "label", "end", -55))
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def range_chart(path: Path, title: str, rows: list[dict[str, object]], keys: tuple[str, str, str], label_key: str, ylabel: str) -> None:
    rows = [row for row in rows if row[keys[1]] != ""]
    if not rows:
        return
    width = max(1000, 80 + 30 * len(rows))
    height = 560
    left, right, top, bottom = 80, 30, 55, 190
    plot_w = width - left - right
    plot_h = height - top - bottom
    values = [float(row[key]) for row in rows for key in keys if row[key] != ""]
    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        min_v -= 1
        max_v += 1
    lines = svg_begin(width, height)
    lines.append(svg_text(width / 2, 25, title, "title", "middle"))
    lines.append(svg_text(16, top + plot_h / 2, ylabel, "label", "middle", -90))
    lines.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    for tick in range(6):
        value = min_v + (max_v - min_v) * tick / 5
        y = top + plot_h - ((value - min_v) / (max_v - min_v)) * plot_h
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        lines.append(svg_text(left - 8, y + 4, f"{value:.2f}", "label", "end"))
    step = plot_w / len(rows)
    for idx, row in enumerate(rows):
        x = left + idx * step + step / 2
        y_low = top + plot_h - ((float(row[keys[0]]) - min_v) / (max_v - min_v)) * plot_h
        y_mid = top + plot_h - ((float(row[keys[1]]) - min_v) / (max_v - min_v)) * plot_h
        y_high = top + plot_h - ((float(row[keys[2]]) - min_v) / (max_v - min_v)) * plot_h
        lines.append(f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="#777" stroke-width="2"/>')
        lines.append(f'<circle cx="{x:.1f}" cy="{y_mid:.1f}" r="4" fill="#3366cc"/>')
        lines.append(svg_text(x, top + plot_h + 12, row[label_key], "label", "end", -55))
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def heatmap(path: Path, title: str, labels: list[str], matrix: list[list[float]], value_format: str = "{:.2f}", reverse: bool = False) -> None:
    if not labels:
        return
    cell = 34
    left, top = 260, 70
    width = left + cell * len(labels) + 40
    height = top + cell * len(labels) + 240
    values = [value for row in matrix for value in row]
    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        max_v = min_v + 1
    lines = svg_begin(width, height)
    lines.append(svg_text(width / 2, 25, title, "title", "middle"))
    for i, label in enumerate(labels):
        lines.append(svg_text(left - 8, top + i * cell + cell * 0.65, label, "label", "end"))
        lines.append(svg_text(left + i * cell + cell * 0.5, top + len(labels) * cell + 8, label, "label", "end", -55))
    for y_idx, row in enumerate(matrix):
        for x_idx, value in enumerate(row):
            ratio = (value - min_v) / (max_v - min_v)
            if reverse:
                ratio = 1 - ratio
            blue = int(245 - ratio * 170)
            red = int(245 - ratio * 35)
            green = int(245 - ratio * 120)
            fill = f"#{red:02x}{green:02x}{blue:02x}"
            x = left + x_idx * cell
            y = top + y_idx * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="white"/>')
            lines.append(svg_text(x + cell / 2, y + cell * 0.62, value_format.format(value), "label", "middle"))
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def make_report(path: Path, summary_rows: list[dict[str, object]], output_dir: Path, max_edit_distance_samples: int) -> None:
    total_files = len(summary_rows)
    total_rows = sum(int(row["rows"]) for row in summary_rows)
    total_accepted = sum(int(row["accepted_rows"]) for row in summary_rows)
    lines = [
        "# InstaNovo Prediction Consistency Plots",
        "",
        f"Prediction files analyzed: {total_files}",
        f"Rows analyzed: {total_rows}",
        f"Rows with non-empty predictions: {total_accepted}",
        "",
        "Exact match, missing-prediction, coverage, score, peptide length, and modification summaries are computed over all rows. Edit-distance distributions use an even deterministic sample of up to "
        f"{max_edit_distance_samples} mismatched predictions per comparison; each CSV records the exact mismatch count and edit-distance sample size.",
        "",
        "## Plots",
        "",
    ]
    for svg in sorted(output_dir.glob("*.svg")):
        lines.append(f"- [{svg.name}]({svg.name})")
    lines.extend(
        [
            "",
            "## Summary Tables",
            "",
        ]
    )
    for csv_file in sorted(output_dir.glob("*.csv")):
        lines.append(f"- [{csv_file.name}]({csv_file.name})")
    path.write_text("\n".join(lines) + "\n")


def sampled_prediction_records(by_config: dict[tuple[str, str], list[Prediction]], max_per_config: int = 3000) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for (input_format, job_id), predictions in sorted(by_config.items()):
        if not predictions:
            continue
        step = max(1, math.ceil(len(predictions) / max_per_config))
        for pred in predictions[::step]:
            records.append(
                {
                    "input_format": input_format,
                    "job_id": job_id,
                    "version": pred.meta.version,
                    "model_family": pred.meta.model_family,
                    "refined": pred.meta.refined,
                    "score": pred.score,
                    "peptide_length": peptide_length(pred.peptide),
                    "modification_count": len(peptide_modifications(pred.peptide)),
                }
            )
    return records


def make_matplotlib_plots(
    output_dir: Path,
    summary_rows: list[dict[str, object]],
    pairwise_rows: list[dict[str, object]],
    format_rows: list[dict[str, object]],
    refinement_rows: list[dict[str, object]],
    beam_rows: list[dict[str, object]],
    format_edit_rows: list[dict[str, object]],
    format_score_rows: list[dict[str, object]],
    refinement_edit_rows: list[dict[str, object]],
    refinement_score_rows: list[dict[str, object]],
    beam_edit_rows: list[dict[str, object]],
    beam_score_rows: list[dict[str, object]],
    by_config: dict[tuple[str, str], list[Prediction]],
    all_mods: Counter[str],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
    except Exception as exc:
        (output_dir / "matplotlib_plots_skipped.txt").write_text(
            f"Matplotlib/seaborn plots were skipped because plotting dependencies are unavailable: {exc}\n"
        )
        return

    sns.set_theme(style="whitegrid")
    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        return
    summary_df["config"] = summary_df["input_format"] + " " + summary_df["job_id"]
    summary_df["coverage"] = pd.to_numeric(summary_df["coverage"])

    def save_current(name: str) -> None:
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=180)
        plt.close()

    plt.figure(figsize=(max(12, len(summary_df) * 0.42), 6))
    sns.barplot(data=summary_df, x="job_id", y="coverage", hue="input_format")
    plt.xticks(rotation=70, ha="right")
    plt.ylim(0, 1.02)
    plt.title("Prediction coverage by input format and configuration")
    save_current("prediction_coverage.png")

    sample_df = pd.DataFrame(sampled_prediction_records(by_config))
    if not sample_df.empty:
        sample_df["score"] = pd.to_numeric(sample_df["score"], errors="coerce")
        sample_df["peptide_length"] = pd.to_numeric(sample_df["peptide_length"], errors="coerce")
        sample_df["config"] = sample_df["input_format"] + " " + sample_df["job_id"]

        score_df = sample_df.dropna(subset=["score"])
        if not score_df.empty:
            plt.figure(figsize=(max(12, score_df["config"].nunique() * 0.42), 6))
            sns.boxplot(data=score_df, x="job_id", y="score", hue="input_format", showfliers=False)
            plt.xticks(rotation=70, ha="right")
            plt.title("Score/confidence distributions by configuration")
            save_current("score_confidence_distributions.png")

        plt.figure(figsize=(max(12, sample_df["config"].nunique() * 0.42), 6))
        sns.boxplot(data=sample_df, x="job_id", y="peptide_length", hue="input_format", showfliers=False)
        plt.xticks(rotation=70, ha="right")
        plt.title("Peptide length distributions by configuration")
        save_current("peptide_length_distributions.png")

    if all_mods:
        mod_df = pd.DataFrame(
            [{"modification": mod, "count": count} for mod, count in all_mods.most_common(25)]
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(data=mod_df, x="count", y="modification", color="#3366cc")
        plt.title("Top modification masses across predictions")
        save_current("modification_frequency.png")

    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        for input_format, sub_df in pairwise_df.groupby("input_format"):
            exact = sub_df.pivot(index="left_job_id", columns="right_job_id", values="exact_match_rate")
            plt.figure(figsize=(max(10, exact.shape[1] * 0.55), max(8, exact.shape[0] * 0.45)))
            sns.heatmap(exact, vmin=0, vmax=1, cmap="Blues", square=True)
            plt.title(f"{input_format.upper()} exact peptide agreement across configurations")
            save_current(f"version_agreement_exact_match_{input_format}.png")

            edit = sub_df.pivot(index="left_job_id", columns="right_job_id", values="mean_mismatch_edit_distance")
            plt.figure(figsize=(max(10, edit.shape[1] * 0.55), max(8, edit.shape[0] * 0.45)))
            sns.heatmap(edit, cmap="mako_r", square=True)
            plt.title(f"{input_format.upper()} mean edit distance for mismatched predictions")
            save_current(f"version_agreement_edit_distance_{input_format}.png")

    format_df = pd.DataFrame(format_rows)
    if not format_df.empty:
        for column in [
            "exact_match_rate",
            "left_missing_rate",
            "right_missing_rate",
            "median_mismatch_edit_distance",
            "median_abs_score_diff",
        ]:
            format_df[column] = pd.to_numeric(format_df[column])

        plt.figure(figsize=(max(12, len(format_df) * 0.45), 6))
        sns.barplot(data=format_df, x="job_id", y="exact_match_rate", color="#3366cc")
        plt.xticks(rotation=70, ha="right")
        plt.ylim(0, 1.02)
        plt.title("MGF vs mzML exact peptide match rate")
        save_current("format_exact_match_rate.png")

        missing_df = format_df.melt(
            id_vars=["job_id"],
            value_vars=["left_missing_rate", "right_missing_rate"],
            var_name="format",
            value_name="missing_rate",
        )
        missing_df["format"] = missing_df["format"].map({"left_missing_rate": "MGF missing", "right_missing_rate": "mzML missing"})
        plt.figure(figsize=(max(12, len(format_df) * 0.45), 6))
        sns.barplot(data=missing_df, x="job_id", y="missing_rate", hue="format")
        plt.xticks(rotation=70, ha="right")
        plt.title("MGF vs mzML missing-prediction rate")
        save_current("format_missing_prediction_rate.png")

        plt.figure(figsize=(max(12, len(format_df) * 0.45), 6))
        sns.barplot(data=format_df, x="job_id", y="median_mismatch_edit_distance", color="#dc3912")
        plt.xticks(rotation=70, ha="right")
        plt.title("MGF vs mzML median edit distance for mismatches")
        save_current("format_mismatch_edit_distance.png")

        plt.figure(figsize=(max(12, len(format_df) * 0.45), 6))
        sns.barplot(data=format_df, x="job_id", y="median_abs_score_diff", color="#109618")
        plt.xticks(rotation=70, ha="right")
        plt.title("MGF vs mzML median absolute score/confidence difference")
        save_current("format_score_difference.png")

        format_edit_df = pd.DataFrame(format_edit_rows)
        if not format_edit_df.empty:
            format_edit_df["edit_distance"] = pd.to_numeric(format_edit_df["edit_distance"])
            plt.figure(figsize=(max(12, format_edit_df["job_id"].nunique() * 0.45), 6))
            sns.boxplot(data=format_edit_df, x="job_id", y="edit_distance", showfliers=False)
            plt.xticks(rotation=70, ha="right")
            plt.title("MGF vs mzML edit-distance distribution for mismatched predictions")
            save_current("format_mismatch_edit_distance_distribution.png")

        format_score_df = pd.DataFrame(format_score_rows)
        if not format_score_df.empty:
            format_score_df["score_diff"] = pd.to_numeric(format_score_df["score_diff"])
            plt.figure(figsize=(max(12, format_score_df["job_id"].nunique() * 0.45), 6))
            sns.boxplot(data=format_score_df, x="job_id", y="score_diff", showfliers=False)
            plt.axhline(0, color="#444", linewidth=1)
            plt.xticks(rotation=70, ha="right")
            plt.title("MGF vs mzML score/log-probability difference distribution")
            save_current("format_score_difference_distribution.png")

        refined_lookup = {
            meta.job_id: "refined" if meta.refined else "not refined"
            for meta in build_metadata().values()
            if meta.input_format == "mgf"
        }
        format_df["refinement"] = format_df["job_id"].map(refined_lookup)
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=format_df, x="refinement", y="exact_match_rate")
        sns.stripplot(data=format_df, x="refinement", y="exact_match_rate", color="#222", alpha=0.65)
        plt.ylim(0, 1.02)
        plt.title("MGF vs mzML agreement for refined and non-refined configurations")
        save_current("format_agreement_refined_vs_not_refined.png")

    refinement_df = pd.DataFrame(refinement_rows)
    if not refinement_df.empty:
        refinement_df["changed_rate"] = 1 - pd.to_numeric(refinement_df["exact_match_rate"])
        refinement_df["median_score_diff"] = pd.to_numeric(refinement_df["median_score_diff"])
        refinement_df["label"] = refinement_df["input_format"] + " " + refinement_df["refined_job_id"]
        plt.figure(figsize=(max(10, len(refinement_df) * 0.65), 6))
        sns.barplot(data=refinement_df, x="label", y="changed_rate", hue="input_format")
        plt.xticks(rotation=60, ha="right")
        plt.title("Prediction changes after InstaNovo+ refinement")
        save_current("refinement_changed_prediction_rate.png")

        plt.figure(figsize=(max(10, len(refinement_df) * 0.65), 6))
        sns.barplot(data=refinement_df, x="label", y="median_score_diff", hue="input_format")
        plt.axhline(0, color="#444", linewidth=1)
        plt.xticks(rotation=60, ha="right")
        plt.title("Median score/confidence shift after refinement")
        save_current("refinement_score_shift.png")

        refinement_edit_df = pd.DataFrame(refinement_edit_rows)
        if not refinement_edit_df.empty:
            refinement_edit_df["edit_distance"] = pd.to_numeric(refinement_edit_df["edit_distance"])
            plt.figure(figsize=(max(10, refinement_edit_df["label"].nunique() * 0.65), 6))
            sns.boxplot(data=refinement_edit_df, x="label", y="edit_distance", hue="input_format", showfliers=False)
            plt.xticks(rotation=60, ha="right")
            plt.title("Sequence edit-distance distribution after refinement")
            save_current("refinement_edit_distance_distribution.png")

        refinement_score_df = pd.DataFrame(refinement_score_rows)
        if not refinement_score_df.empty:
            refinement_score_df["score_diff"] = pd.to_numeric(refinement_score_df["score_diff"])
            plt.figure(figsize=(max(10, refinement_score_df["label"].nunique() * 0.65), 6))
            sns.boxplot(data=refinement_score_df, x="label", y="score_diff", hue="input_format", showfliers=False)
            plt.axhline(0, color="#444", linewidth=1)
            plt.xticks(rotation=60, ha="right")
            plt.title("Score/log-probability shift distribution after refinement")
            save_current("refinement_score_shift_distribution.png")

    beam_df = pd.DataFrame(beam_rows)
    if not beam_df.empty:
        beam_df["changed_rate"] = 1 - pd.to_numeric(beam_df["exact_match_rate"])
        beam_df["label"] = beam_df["input_format"] + " " + beam_df["beam_job_id"]
        plt.figure(figsize=(max(10, len(beam_df) * 0.65), 6))
        sns.barplot(data=beam_df, x="label", y="changed_rate", hue="input_format")
        plt.xticks(rotation=60, ha="right")
        plt.title("Prediction changes for beam search vs greedy")
        save_current("beam_vs_greedy_changed_prediction_rate.png")

        beam_edit_df = pd.DataFrame(beam_edit_rows)
        if not beam_edit_df.empty:
            beam_edit_df["edit_distance"] = pd.to_numeric(beam_edit_df["edit_distance"])
            plt.figure(figsize=(max(10, beam_edit_df["label"].nunique() * 0.65), 6))
            sns.boxplot(data=beam_edit_df, x="label", y="edit_distance", hue="input_format", showfliers=False)
            plt.xticks(rotation=60, ha="right")
            plt.title("Sequence edit-distance distribution for beam search vs greedy")
            save_current("beam_vs_greedy_edit_distance_distribution.png")

        beam_score_df = pd.DataFrame(beam_score_rows)
        if not beam_score_df.empty:
            beam_score_df["score_diff"] = pd.to_numeric(beam_score_df["score_diff"])
            plt.figure(figsize=(max(10, beam_score_df["label"].nunique() * 0.65), 6))
            sns.boxplot(data=beam_score_df, x="label", y="score_diff", hue="input_format", showfliers=False)
            plt.axhline(0, color="#444", linewidth=1)
            plt.xticks(rotation=60, ha="right")
            plt.title("Score/log-probability difference distribution for beam search vs greedy")
            save_current("beam_vs_greedy_score_difference_distribution.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate InstaNovo prediction behavior and format consistency plots.")
    parser.add_argument("predictions_dir", type=Path, help="Directory containing full/mgf and full/mzml prediction CSVs.")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pdv-instanovo-plots"), help="Directory for CSV summaries and SVG plots.")
    parser.add_argument(
        "--max-edit-distance-samples",
        type=int,
        default=2000,
        help="Maximum mismatched predictions per comparison used for edit-distance statistics. Exact match and missing rates are always exact.",
    )
    args = parser.parse_args()

    predictions_dir = args.predictions_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata()

    by_config: dict[tuple[str, str], list[Prediction]] = {}
    source_rows: dict[tuple[str, str], int] = {}
    summary_rows: list[dict[str, object]] = []
    all_mods: Counter[str] = Counter()

    files = sorted((predictions_dir / "full").glob("*/*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction CSV files found under {predictions_dir / 'full'}")

    for path in files:
        meta = metadata.get(path.name)
        if meta is None:
            continue
        predictions, rows = read_prediction_file(path, meta)
        key = (meta.input_format, meta.job_id)
        by_config[key] = predictions
        source_rows[key] = rows
        scores = [pred.score for pred in predictions if pred.score is not None]
        lengths = [peptide_length(pred.peptide) for pred in predictions]
        mods = [mod for pred in predictions for mod in peptide_modifications(pred.peptide)]
        all_mods.update(mods)
        label = f"{meta.input_format} {meta.job_id}"
        summary_rows.append(
            {
                "input_format": meta.input_format,
                "version": meta.version,
                "job_id": meta.job_id,
                "model_family": meta.model_family,
                "method": meta.method,
                "refined": meta.refined,
                "beam_count": meta.beam_count or "",
                "output_file": meta.output_file,
                "rows": rows,
                "accepted_rows": len(predictions),
                "coverage": len(predictions) / rows if rows else 0.0,
                "score_p10": quantile(scores, 0.1) if scores else "",
                "score_median": quantile(scores, 0.5) if scores else "",
                "score_p90": quantile(scores, 0.9) if scores else "",
                "length_p10": quantile(lengths, 0.1) if lengths else "",
                "length_median": quantile(lengths, 0.5) if lengths else "",
                "length_p90": quantile(lengths, 0.9) if lengths else "",
                "modified_predictions": sum(1 for pred in predictions if peptide_modifications(pred.peptide)),
                "label": label,
            }
        )

    if not summary_rows:
        raise RuntimeError(
            f"No prediction files under {predictions_dir / 'full'} matched the expected full-run filenames."
        )
    if args.max_edit_distance_samples < 1:
        raise ValueError("--max-edit-distance-samples must be at least 1")

    summary_fields = [
        "input_format",
        "version",
        "job_id",
        "model_family",
        "method",
        "refined",
        "beam_count",
        "output_file",
        "rows",
        "accepted_rows",
        "coverage",
        "score_p10",
        "score_median",
        "score_p90",
        "length_p10",
        "length_median",
        "length_p90",
        "modified_predictions",
        "label",
    ]
    write_csv(output_dir / "prediction_summary.csv", summary_rows, summary_fields)

    bar_chart(
        output_dir / "prediction_coverage.svg",
        "Prediction coverage by input format and configuration",
        summary_rows,
        "coverage",
        "label",
        "input_format",
        "Coverage",
    )
    range_chart(
        output_dir / "score_distribution_summary.svg",
        "Score/confidence distribution summary (p10, median, p90)",
        summary_rows,
        ("score_p10", "score_median", "score_p90"),
        "label",
        "Score",
    )
    range_chart(
        output_dir / "peptide_length_distribution_summary.svg",
        "Peptide length distribution summary (p10, median, p90)",
        summary_rows,
        ("length_p10", "length_median", "length_p90"),
        "label",
        "Peptide length",
    )

    mod_rows = [
        {"modification": mod, "count": count}
        for mod, count in all_mods.most_common(25)
    ]
    write_csv(output_dir / "modification_frequency.csv", mod_rows, ["modification", "count"])
    bar_chart(
        output_dir / "modification_frequency.svg",
        "Top modification masses across predictions",
        mod_rows,
        "count",
        "modification",
        None,
        "Count",
    )

    pairwise_rows: list[dict[str, object]] = []
    for input_format in sorted({key[0] for key in by_config}):
        keys = sorted([key for key in by_config if key[0] == input_format], key=lambda key: key[1])
        prediction_maps = {key: {pred.scan: pred for pred in by_config[key]} for key in keys}
        labels = [key[1] for key in keys]
        exact_matrix: list[list[float]] = []
        edit_matrix: list[list[float]] = []
        for left_key in keys:
            exact_row: list[float] = []
            edit_row: list[float] = []
            left_map = prediction_maps[left_key]
            for right_key in keys:
                right_map = prediction_maps[right_key]
                stats = compare_prediction_maps(left_map, right_map, args.max_edit_distance_samples)
                exact_row.append(float(stats["exact_match_rate"]))
                edit_row.append(float(stats["mean_mismatch_edit_distance"]))
                pairwise_rows.append(
                    {
                        "input_format": input_format,
                        "left_job_id": left_key[1],
                        "right_job_id": right_key[1],
                        **stats,
                    }
                )
            exact_matrix.append(exact_row)
            edit_matrix.append(edit_row)
        heatmap(
            output_dir / f"version_agreement_exact_match_{input_format}.svg",
            f"{input_format.upper()} exact peptide agreement across configurations",
            labels,
            exact_matrix,
            "{:.2f}",
        )
        heatmap(
            output_dir / f"version_agreement_edit_distance_{input_format}.svg",
            f"{input_format.upper()} mean edit distance for mismatched predictions",
            labels,
            edit_matrix,
            "{:.1f}",
            reverse=True,
        )
    write_csv(
        output_dir / "version_agreement_pairwise.csv",
        pairwise_rows,
        [
            "input_format",
            "left_job_id",
            "right_job_id",
            "total_scans",
            "common_predicted",
            "exact_matches",
            "exact_match_rate",
            "left_missing",
            "right_missing",
            "left_missing_rate",
            "right_missing_rate",
            "mismatch_count",
            "edit_distance_sample_size",
            "mean_mismatch_edit_distance",
            "median_mismatch_edit_distance",
            "median_score_diff",
            "median_abs_score_diff",
            "p90_abs_score_diff",
        ],
    )

    format_rows: list[dict[str, object]] = []
    format_edit_rows: list[dict[str, object]] = []
    format_score_rows: list[dict[str, object]] = []
    for job_id in sorted({key[1] for key in by_config}):
        mgf = {pred.scan: pred for pred in by_config.get(("mgf", job_id), [])}
        mzml = {pred.scan: pred for pred in by_config.get(("mzml", job_id), [])}
        if not mgf or not mzml:
            continue
        format_rows.append({"job_id": job_id, **compare_prediction_maps(mgf, mzml, args.max_edit_distance_samples)})
        edit_distances, score_diffs = comparison_samples(mgf, mzml, args.max_edit_distance_samples)
        format_edit_rows.extend({"job_id": job_id, "edit_distance": value} for value in edit_distances)
        format_score_rows.extend({"job_id": job_id, "score_diff": value, "abs_score_diff": abs(value)} for value in score_diffs)
    write_csv(
        output_dir / "format_consistency.csv",
        format_rows,
        [
            "job_id",
            "total_scans",
            "common_predicted",
            "exact_matches",
            "exact_match_rate",
            "left_missing",
            "right_missing",
            "left_missing_rate",
            "right_missing_rate",
            "mismatch_count",
            "edit_distance_sample_size",
            "mean_mismatch_edit_distance",
            "median_mismatch_edit_distance",
            "median_score_diff",
            "median_abs_score_diff",
            "p90_abs_score_diff",
        ],
    )
    write_csv(output_dir / "format_mismatch_edit_distance_samples.csv", format_edit_rows, ["job_id", "edit_distance"])
    write_csv(output_dir / "format_score_difference_samples.csv", format_score_rows, ["job_id", "score_diff", "abs_score_diff"])
    bar_chart(
        output_dir / "format_exact_match_rate.svg",
        "MGF vs mzML exact peptide match rate by configuration",
        format_rows,
        "exact_match_rate",
        "job_id",
        None,
        "Exact match rate",
    )
    missing_rows = []
    for row in format_rows:
        missing_rows.append({"job_id": row["job_id"] + " mgf-missing", "missing_rate": row["left_missing_rate"]})
        missing_rows.append({"job_id": row["job_id"] + " mzml-missing", "missing_rate": row["right_missing_rate"]})
    bar_chart(
        output_dir / "format_missing_prediction_rate.svg",
        "MGF vs mzML missing-prediction rate by configuration",
        missing_rows,
        "missing_rate",
        "job_id",
        None,
        "Missing rate",
    )
    bar_chart(
        output_dir / "format_mismatch_edit_distance.svg",
        "MGF vs mzML median edit distance for mismatched predictions",
        format_rows,
        "median_mismatch_edit_distance",
        "job_id",
        None,
        "Median edit distance",
    )
    bar_chart(
        output_dir / "format_score_difference.svg",
        "MGF vs mzML median absolute score difference by configuration",
        format_rows,
        "median_abs_score_diff",
        "job_id",
        None,
        "Median abs score difference",
    )
    refined_format_rows = []
    for row in format_rows:
        meta = next((meta for meta in metadata.values() if meta.job_id == row["job_id"] and meta.input_format == "mgf"), None)
        if meta is not None:
            refined_format_rows.append({"job_id": row["job_id"], "exact_match_rate": row["exact_match_rate"], "refined": "refined" if meta.refined else "not refined"})
    bar_chart(
        output_dir / "format_agreement_refined_vs_not_refined.svg",
        "MGF vs mzML agreement for refined and non-refined configurations",
        refined_format_rows,
        "exact_match_rate",
        "job_id",
        "refined",
        "Exact match rate",
    )

    refinement_pairs = {
        "1.1.0-plus-refined": "1.1.0-transformer-greedy",
        "1.1.3-plus-refined": "1.1.3-transformer-greedy",
        "1.1.3-plus-refined-verbose": "1.1.3-transformer-greedy",
        "1.2.2-combined-refined": "1.2.2-transformer-beams5",
    }
    refinement_rows: list[dict[str, object]] = []
    refinement_edit_rows: list[dict[str, object]] = []
    refinement_score_rows: list[dict[str, object]] = []
    for input_format in ("mgf", "mzml"):
        for refined_job, base_job in refinement_pairs.items():
            refined = {pred.scan: pred for pred in by_config.get((input_format, refined_job), [])}
            base = {pred.scan: pred for pred in by_config.get((input_format, base_job), [])}
            if not refined or not base:
                continue
            stats = compare_prediction_maps(base, refined, args.max_edit_distance_samples)
            refinement_rows.append({"input_format": input_format, "base_job_id": base_job, "refined_job_id": refined_job, **stats})
            edit_distances, score_diffs = comparison_samples(base, refined, args.max_edit_distance_samples)
            label = f"{input_format} {refined_job}"
            refinement_edit_rows.extend(
                {
                    "input_format": input_format,
                    "base_job_id": base_job,
                    "refined_job_id": refined_job,
                    "label": label,
                    "edit_distance": value,
                }
                for value in edit_distances
            )
            refinement_score_rows.extend(
                {
                    "input_format": input_format,
                    "base_job_id": base_job,
                    "refined_job_id": refined_job,
                    "label": label,
                    "score_diff": value,
                    "abs_score_diff": abs(value),
                }
                for value in score_diffs
            )
    write_csv(
        output_dir / "refinement_effects.csv",
        refinement_rows,
        [
            "input_format",
            "base_job_id",
            "refined_job_id",
            "total_scans",
            "common_predicted",
            "exact_matches",
            "exact_match_rate",
            "left_missing",
            "right_missing",
            "left_missing_rate",
            "right_missing_rate",
            "mismatch_count",
            "edit_distance_sample_size",
            "mean_mismatch_edit_distance",
            "median_mismatch_edit_distance",
            "median_score_diff",
            "median_abs_score_diff",
            "p90_abs_score_diff",
        ],
    )
    write_csv(
        output_dir / "refinement_edit_distance_samples.csv",
        refinement_edit_rows,
        ["input_format", "base_job_id", "refined_job_id", "label", "edit_distance"],
    )
    write_csv(
        output_dir / "refinement_score_difference_samples.csv",
        refinement_score_rows,
        ["input_format", "base_job_id", "refined_job_id", "label", "score_diff", "abs_score_diff"],
    )
    refinement_plot_rows = [
        {"label": f"{row['input_format']} {row['refined_job_id']}", "changed_rate": 1 - float(row["exact_match_rate"])}
        for row in refinement_rows
    ]
    bar_chart(
        output_dir / "refinement_changed_prediction_rate.svg",
        "Prediction changes after InstaNovo+ refinement",
        refinement_plot_rows,
        "changed_rate",
        "label",
        None,
        "Changed rate",
    )

    beam_pairs = {
        "1.0.0-transformer-beams5": "1.0.0-transformer-greedy",
        "1.1.0-transformer-beams5": "1.1.0-transformer-greedy",
        "1.1.3-transformer-beams5": "1.1.3-transformer-greedy",
        "1.2.2-transformer-beams5": "1.2.2-transformer-greedy",
    }
    beam_rows: list[dict[str, object]] = []
    beam_edit_rows: list[dict[str, object]] = []
    beam_score_rows: list[dict[str, object]] = []
    for input_format in ("mgf", "mzml"):
        for beam_job, greedy_job in beam_pairs.items():
            beam = {pred.scan: pred for pred in by_config.get((input_format, beam_job), [])}
            greedy = {pred.scan: pred for pred in by_config.get((input_format, greedy_job), [])}
            if not beam or not greedy:
                continue
            stats = compare_prediction_maps(greedy, beam, args.max_edit_distance_samples)
            beam_rows.append({"input_format": input_format, "greedy_job_id": greedy_job, "beam_job_id": beam_job, **stats})
            edit_distances, score_diffs = comparison_samples(greedy, beam, args.max_edit_distance_samples)
            label = f"{input_format} {beam_job}"
            beam_edit_rows.extend(
                {
                    "input_format": input_format,
                    "greedy_job_id": greedy_job,
                    "beam_job_id": beam_job,
                    "label": label,
                    "edit_distance": value,
                }
                for value in edit_distances
            )
            beam_score_rows.extend(
                {
                    "input_format": input_format,
                    "greedy_job_id": greedy_job,
                    "beam_job_id": beam_job,
                    "label": label,
                    "score_diff": value,
                    "abs_score_diff": abs(value),
                }
                for value in score_diffs
            )
    write_csv(
        output_dir / "beam_vs_greedy.csv",
        beam_rows,
        [
            "input_format",
            "greedy_job_id",
            "beam_job_id",
            "total_scans",
            "common_predicted",
            "exact_matches",
            "exact_match_rate",
            "left_missing",
            "right_missing",
            "left_missing_rate",
            "right_missing_rate",
            "mismatch_count",
            "edit_distance_sample_size",
            "mean_mismatch_edit_distance",
            "median_mismatch_edit_distance",
            "median_score_diff",
            "median_abs_score_diff",
            "p90_abs_score_diff",
        ],
    )
    write_csv(
        output_dir / "beam_vs_greedy_edit_distance_samples.csv",
        beam_edit_rows,
        ["input_format", "greedy_job_id", "beam_job_id", "label", "edit_distance"],
    )
    write_csv(
        output_dir / "beam_vs_greedy_score_difference_samples.csv",
        beam_score_rows,
        ["input_format", "greedy_job_id", "beam_job_id", "label", "score_diff", "abs_score_diff"],
    )
    beam_plot_rows = [
        {"label": f"{row['input_format']} {row['beam_job_id']}", "changed_rate": 1 - float(row["exact_match_rate"])}
        for row in beam_rows
    ]
    bar_chart(
        output_dir / "beam_vs_greedy_changed_prediction_rate.svg",
        "Prediction changes for beam search vs greedy",
        beam_plot_rows,
        "changed_rate",
        "label",
        None,
        "Changed rate",
    )

    make_report(output_dir / "README.md", summary_rows, output_dir, args.max_edit_distance_samples)
    make_matplotlib_plots(
        output_dir,
        summary_rows,
        pairwise_rows,
        format_rows,
        refinement_rows,
        beam_rows,
        format_edit_rows,
        format_score_rows,
        refinement_edit_rows,
        refinement_score_rows,
        beam_edit_rows,
        beam_score_rows,
        by_config,
        all_mods,
    )
    print(f"Wrote plots and summaries to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
