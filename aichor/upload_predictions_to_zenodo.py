#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "aichor" / "generate_instanovo_predictions.py"
DEFAULT_TITLE = "PDV InstaNovo v1+ prediction sample files"
DEFAULT_DESCRIPTION = (
    "InstaNovo prediction CSV outputs generated from the MGF and mzML files "
    "for validating PDV InstaNovo import support."
)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def load_runner_module():
    spec = importlib.util.spec_from_file_location("pdv_instanovo_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_command(command: list[str], source_path: Path, output_path: Path) -> str:
    displayed = []
    for token in command:
        text = str(token)
        if text == str(source_path):
            text = "<input>"
        elif text == str(output_path):
            text = "<output>"
        elif text.startswith("data_path="):
            text = "data_path=<input>"
        elif text.startswith("output_path="):
            text = "output_path=<output>"
        elif text.startswith("instanovo_predictions_path="):
            text = "instanovo_predictions_path=<transformer-greedy-output>"
        elif text.startswith("/opt/instanovo-env/bin/"):
            text = Path(text).name
        displayed.append(text)
    return shlex.join(displayed)


def command_value(command: list[str], flag: str) -> str | None:
    try:
        index = command.index(flag)
    except ValueError:
        return None
    next_index = index + 1
    if next_index >= len(command):
        return None
    return command[next_index]


def command_assignment(command: list[str], name: str) -> str | None:
    prefix = f"{name}="
    for token in command:
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def checkpoint_version(value: str | None, prefix: str) -> str:
    if not value:
        return "-"
    name = Path(value).name
    if name.endswith(".ckpt"):
        name = name[:-5]
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def instanovo_checkpoint_version(job) -> str:
    checkpoint = command_value(job.command, "-i") or command_assignment(job.command, "model_path")
    if checkpoint:
        return checkpoint_version(checkpoint, "instanovo-")
    if job.depends_on is not None:
        match = re.search(r"model-instanovo-(v[0-9]+(?:[.][0-9]+)*(?:-[A-Za-z0-9.]+)?)", job.depends_on.name)
        if match:
            return match.group(1)
    return "-"


def instanovo_software_version(job_id: str, package_version: str) -> str:
    return package_version if "transformer" in job_id or "combined" in job_id or "plus-refined" in job_id else "-"


def instanovoplus_software_version(job_id: str, package_version: str) -> str:
    return package_version if "plus" in job_id or "combined" in job_id else "-"


def prediction_mode(job_id: str) -> str:
    if "combined" in job_id:
        return "Transformer + InstaNovo+ refinement"
    if "plus-refined" in job_id:
        return "InstaNovo+ refinement"
    if "plus-standalone" in job_id:
        return "InstaNovo+ no refinement"
    if "beams" in job_id:
        return "Transformer beam search"
    if "greedy" in job_id:
        return "Transformer greedy"
    return job_id


def search_method(job_id: str) -> str:
    if "beams" in job_id or "combined" in job_id:
        return "beam search"
    if "greedy" in job_id or "plus-refined" in job_id:
        return "greedy"
    return "-"


def beam_count(job_id: str) -> str:
    if "beams5" in job_id or "combined" in job_id:
        return "5"
    if "greedy" in job_id or "plus-refined" in job_id:
        return "1"
    return "-"


def prediction_table() -> str:
    runner = load_runner_module()
    rows = [
        "| Input | Prediction file | InstaNovo software version | InstaNovo+ software version | InstaNovo checkpoint version | InstaNovo+ checkpoint version | Prediction mode | Search method | Beams | Command / flags |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    sources = [
        runner.SourceFile("full", "mgf", Path("/workspace/work/SF_200217_U2OS_TiO2_HCD_OT_rep1.full.mgf")),
        runner.SourceFile("full", "mzML", Path("/workspace/work/SF_200217_U2OS_TiO2_HCD_OT_rep1.full.mzML")),
    ]
    for source in sources:
        for job in runner.build_jobs(source):
            instanovo_checkpoint = instanovo_checkpoint_version(job)
            instanovoplus_checkpoint = checkpoint_version(
                command_value(job.command, "-p"),
                "instanovoplus-",
            )
            rows.append(
                "| "
                + " | ".join(
                    [
                        source.source_format,
                        f"`{job.output_path.name}`",
                        instanovo_software_version(job.job_id, job.version),
                        instanovoplus_software_version(job.job_id, job.version),
                        instanovo_checkpoint,
                        instanovoplus_checkpoint,
                        prediction_mode(job.job_id),
                        search_method(job.job_id),
                        beam_count(job.job_id),
                        f"`{display_command(job.command, source.path, job.output_path)}`",
                    ]
                )
                + " |"
            )
    return "\n".join(rows)


def write_readme(output_dir: Path) -> Path:
    runner = load_runner_module()
    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# PDV InstaNovo v1+ Prediction Sample Files",
                "",
                "This dataset contains InstaNovo prediction CSV outputs generated from the MGF and mzML files for validating PDV InstaNovo import support.",
                "",
                "The outputs were generated for both input file formats used for validation.",
                "",
                "## Input Files",
                "",
                "| Format | Source URL |",
                "| --- | --- |",
                f"| MGF | `{runner.MGF_URL}` |",
                f"| mzML | `{runner.MZML_URL}` |",
                "",
                "The `.gz` inputs were downloaded from the URLs above and decompressed before running InstaNovo.",
                "",
                "## Generation",
                "",
                "- Predictions were run on a `NVIDIA H100 80GB HBM3` GPU.",
                "",
                "The command table uses `<input>` and `<output>` placeholders for the concrete input and output paths inside the prediction runtime.",
                "",
                "## Prediction Files and Commands",
                "",
                prediction_table(),
                "",
                "## Files",
                "",
                "- `*.csv`: individual full prediction CSV outputs. File names include the input format, InstaNovo package version, model/checkpoint, and prediction/refinement options.",
                "- `prediction_manifest.csv` and `prediction_manifest.json`: prediction runner output manifests.",
                "- `SHA256SUMS`: checksums for uploaded files.",
                "",
            ]
        )
    )
    return readme


def copy_manifest_files(predictions_dir: Path, output_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for name in ("prediction_manifest.csv", "prediction_manifest.json"):
        source = predictions_dir / name
        if source.exists():
            target = output_dir / name
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def copy_prediction_csvs(predictions_dir: Path, output_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for source_format in ("mgf", "mzml"):
        source_dir = predictions_dir / "full" / source_format
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing prediction directory: {source_dir}")
        for source in sorted(source_dir.glob("*.csv")):
            target = output_dir / source.name
            target.write_bytes(source.read_bytes())
            copied.append(target)
    return copied


def write_checksums(paths: list[Path], output_dir: Path) -> Path:
    checksum_path = output_dir / "SHA256SUMS"
    with checksum_path.open("w") as handle:
        for path in sorted(paths, key=lambda p: p.name):
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    return checksum_path


def prepare_upload_files(predictions_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    readme = write_readme(output_dir)
    manifests = copy_manifest_files(predictions_dir, output_dir)
    prediction_csvs = copy_prediction_csvs(predictions_dir, output_dir)
    checksum = write_checksums([readme, *manifests, *prediction_csvs], output_dir)
    return [readme, *manifests, *prediction_csvs, checksum]


def zenodo_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def create_deposition(base_url: str, token: str, title: str, creator: str | None) -> dict:
    metadata = {
        "title": title,
        "upload_type": "dataset",
        "description": DEFAULT_DESCRIPTION,
    }
    if creator:
        metadata["creators"] = [{"name": creator}]
    payload = {"metadata": metadata}
    response = requests.post(
        f"{base_url.rstrip('/')}/api/deposit/depositions",
        headers={**zenodo_headers(token), "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def get_deposition(base_url: str, token: str, deposition_id: str) -> dict:
    response = requests.get(
        f"{base_url.rstrip('/')}/api/deposit/depositions/{deposition_id}",
        headers=zenodo_headers(token),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def upload_file(bucket_url: str, token: str, path: Path) -> None:
    upload_url = f"{bucket_url.rstrip('/')}/{quote(path.name)}"
    print(f"Uploading {path.name} ({path.stat().st_size} bytes)", flush=True)
    with path.open("rb") as handle:
        response = requests.put(
            upload_url,
            params={"access_token": token},
            data=handle,
            timeout=None,
        )
    response.raise_for_status()


def upload_to_zenodo(
    files: list[Path],
    output_dir: Path,
    token: str,
    base_url: str,
    deposition_id: str | None,
    title: str,
    creator: str | None,
) -> dict:
    deposition = get_deposition(base_url, token, deposition_id) if deposition_id else create_deposition(base_url, token, title, creator)
    bucket_url = deposition.get("links", {}).get("bucket")
    if not bucket_url:
        raise RuntimeError("Zenodo deposition response did not include links.bucket")

    for path in files:
        upload_file(bucket_url, token, path)

    deposition = get_deposition(base_url, token, str(deposition["id"]))
    (output_dir / "zenodo_deposition.json").write_text(json.dumps(deposition, indent=2))
    return deposition


def main() -> int:
    parser = argparse.ArgumentParser(description="Package full InstaNovo predictions and upload them to a Zenodo draft.")
    parser.add_argument("predictions_dir", type=Path, help="Downloaded full predictions directory containing full/mgf and full/mzml.")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/pdv-zenodo-instanovo"), help="Local bundle directory.")
    parser.add_argument("--zenodo-url", default=os.environ.get("ZENODO_URL", "https://zenodo.org"), help="Zenodo base URL.")
    parser.add_argument("--deposition-id", default=os.environ.get("ZENODO_DEPOSITION_ID"), help="Existing draft deposition ID to upload to.")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Zenodo draft title when creating a new deposition.")
    parser.add_argument("--creator", default=os.environ.get("ZENODO_CREATOR", "BioGeek"), help="Creator name for draft metadata.")
    parser.add_argument("--no-upload", action="store_true", help="Only package files; do not create or upload to Zenodo.")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    predictions_dir = args.predictions_dir.resolve()
    if not (predictions_dir / "full" / "mgf").is_dir() or not (predictions_dir / "full" / "mzml").is_dir():
        raise FileNotFoundError("predictions_dir must contain full/mgf and full/mzml directories")

    upload_files = prepare_upload_files(predictions_dir, args.output_dir.resolve())
    print("Prepared upload files:")
    for path in upload_files:
        print(f"  {path}")

    if args.no_upload:
        return 0

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise RuntimeError("ZENODO_TOKEN is not set in the environment or .env")

    deposition = upload_to_zenodo(
        upload_files,
        args.output_dir.resolve(),
        token,
        args.zenodo_url,
        args.deposition_id,
        args.title,
        args.creator,
    )
    print("Zenodo draft uploaded.")
    print(f"Deposition ID: {deposition['id']}")
    print(f"Draft URL: {deposition.get('links', {}).get('html', 'unknown')}")
    print("The draft has not been published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
