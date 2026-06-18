#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path("/workspace")
INPUT_DIR = ROOT / "inputs"
WORK_DIR = ROOT / "work"
OUT_DIR = ROOT / "predictions"
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"
INOVO_ROOT = Path(os.environ.get("INOVO_ROOT", "/opt/instanovo-versions"))
PYTHON = Path(os.environ.get("INOVO_VENV", "/opt/instanovo-env")) / "bin" / "python"
INSTANOVO = Path(os.environ.get("INOVO_VENV", "/opt/instanovo-env")) / "bin" / "instanovo"


MGF_URL = os.environ.get(
    "PDV_INSTANOVO_MGF_URL",
    "http://pdv.zhang-lab.org/data/download/test_data/msdata/SF_200217_U2OS_TiO2_HCD_OT_rep1.mgf.gz",
)
MZML_URL = os.environ.get(
    "PDV_INSTANOVO_MZML_URL",
    "http://pdv.zhang-lab.org/data/download/test_data/msdata/SF_200217_U2OS_TiO2_HCD_OT_rep1.mzML.gz",
)
MAX_SPECTRA = int(os.environ.get("PDV_INSTANOVO_MAX_SPECTRA", "10"))
BATCH_SIZE = os.environ.get("PDV_INSTANOVO_BATCH_SIZE", "512")
NUM_WORKERS = os.environ.get("PDV_INSTANOVO_NUM_WORKERS", "4")
CONTINUE_ON_ERROR = os.environ.get("PDV_INSTANOVO_CONTINUE_ON_ERROR", "true").lower() == "true"
UPLOAD_PREFIX = os.environ.get("PDV_INSTANOVO_UPLOAD_PREFIX", "instanovo-pdv-prediction-samples")


@dataclass(frozen=True)
class SourceFile:
    label: str
    source_format: str
    path: Path


@dataclass(frozen=True)
class PredictionJob:
    job_id: str
    version: str
    command: list[str]
    output_path: Path
    schema_group: str
    depends_on: Path | None = None


def log(message: str) -> None:
    print(f"[pdv-instanovo] {message}", flush=True)


def ensure_dirs() -> None:
    for path in (INPUT_DIR, WORK_DIR, OUT_DIR, LOG_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def download_and_gunzip(url: str, target: Path) -> Path:
    gz_path = target.with_suffix(target.suffix + ".gz")
    if not gz_path.exists():
        log(f"Downloading {url} -> {gz_path}")
        urlretrieve(url, gz_path)
    else:
        log(f"Using existing download {gz_path}")

    if not target.exists():
        log(f"Decompressing {gz_path} -> {target}")
        with gzip.open(gz_path, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        log(f"Using existing decompressed file {target}")
    return target


def subset_mgf(source: Path, target: Path, max_spectra: int) -> Path:
    if max_spectra <= 0:
        return source
    if target.exists():
        return target

    count = 0
    in_block = False
    with source.open("rt", errors="replace") as src, target.open("wt") as dst:
        for line in src:
            if line.strip() == "BEGIN IONS":
                if count >= max_spectra:
                    break
                in_block = True
            if in_block:
                dst.write(line)
            if line.strip() == "END IONS" and in_block:
                count += 1
                in_block = False
    log(f"Wrote first {count} MGF spectra to {target}")
    return target


def subset_mzml(source: Path, target: Path, max_spectra: int) -> Path:
    if max_spectra <= 0:
        return source
    if target.exists():
        return target

    import pyopenms as oms

    exp = oms.MSExperiment()
    oms.MzMLFile().load(str(source), exp)
    subset = oms.MSExperiment()
    for spectrum in exp.getSpectra()[:max_spectra]:
        subset.addSpectrum(spectrum)
    oms.MzMLFile().store(str(target), subset)
    log(f"Wrote first {subset.size()} mzML spectra to {target}")
    return target


def prepare_inputs() -> list[SourceFile]:
    mgf = download_and_gunzip(MGF_URL, INPUT_DIR / "SF_200217_U2OS_TiO2_HCD_OT_rep1.mgf")
    mzml = download_and_gunzip(MZML_URL, INPUT_DIR / "SF_200217_U2OS_TiO2_HCD_OT_rep1.mzML")

    suffix = f"first{MAX_SPECTRA}" if MAX_SPECTRA > 0 else "full"
    mgf_input = subset_mgf(mgf, WORK_DIR / f"SF_200217_U2OS_TiO2_HCD_OT_rep1.{suffix}.mgf", MAX_SPECTRA)
    mzml_input = subset_mzml(mzml, WORK_DIR / f"SF_200217_U2OS_TiO2_HCD_OT_rep1.{suffix}.mzML", MAX_SPECTRA)

    return [
        SourceFile(suffix, "mgf", mgf_input),
        SourceFile(suffix, "mzML", mzml_input),
    ]


def run_env(version: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(INOVO_ROOT / version)
    env["PATH"] = f"{PYTHON.parent}:{env.get('PATH', '')}"
    env["MPLCONFIGDIR"] = str(WORK_DIR / "mpl")
    # Prevent old InstaNovo S3 helpers from auto-uploading local intermediate paths.
    env.pop("AICHOR_LOGS_PATH", None)
    return env


def upload_destination_root() -> str | None:
    output_root = os.environ.get("AICHOR_OUTPUT_PATH")
    if not output_root:
        return None
    return output_root.rstrip("/") + "/" + UPLOAD_PREFIX.strip("/")


@lru_cache(maxsize=1)
def s3_filesystem():
    import s3fs

    endpoint = os.environ.get("S3_ENDPOINT")
    return s3fs.S3FileSystem(client_kwargs={"endpoint_url": endpoint} if endpoint else None)


def remote_path(path: Path) -> str | None:
    destination = upload_destination_root()
    if not destination:
        return None
    return destination + "/" + str(path.relative_to(ROOT))


def remote_file_size(path: Path) -> int | None:
    remote = remote_path(path)
    if remote is None:
        return None

    try:
        info = s3_filesystem().info(remote)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if "not found" in str(exc).lower() or "no such" in str(exc).lower():
            return None
        raise

    size = info.get("size", info.get("Size"))
    return int(size) if size is not None else 0


def download_remote_file(path: Path) -> bool:
    size = remote_file_size(path)
    if not size:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    remote = remote_path(path)
    assert remote is not None
    log(f"Downloading existing S3 output {remote} -> {path}")
    s3_filesystem().get(remote, str(path))
    return True


def upload_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return

    remote = remote_path(path)
    if remote is None:
        return

    log(f"Upload {path} -> {remote}")
    s3_filesystem().put(str(path), remote)


def out_path(source: SourceFile, filename: str) -> Path:
    return OUT_DIR / source.label / source.source_format.lower() / filename


def build_jobs(source: SourceFile) -> list[PredictionJob]:
    base = f"SF_200217_U2OS_TiO2_HCD_OT_rep1.{source.label}.{source.source_format.lower()}"

    v100_greedy = out_path(
        source,
        f"{base}.instanovo-1.0.0.transformer.model-instanovo-v1.0.0.denovo.greedy.beams-1.csv",
    )
    v100_beams = out_path(
        source,
        f"{base}.instanovo-1.0.0.transformer.model-instanovo-v1.0.0.denovo.beam-search.beams-5.save-beams.csv",
    )
    v110_greedy = out_path(
        source,
        f"{base}.instanovo-1.1.0.transformer.model-instanovo-v1.1.0.denovo.greedy.beams-1.columns-predictions.csv",
    )
    v110_beams = out_path(
        source,
        f"{base}.instanovo-1.1.0.transformer.model-instanovo-v1.1.0.denovo.beam-search.beams-5.save-beams.columns-predictions.csv",
    )
    v110_plus = out_path(
        source,
        f"{base}.instanovo-1.1.0.instanovoplus.model-instanovoplus-v1.1.0-alpha.denovo.no-refinement.n-preds-1.csv",
    )
    v110_refined = out_path(
        source,
        f"{base}.instanovo-1.1.0.instanovoplus.model-instanovoplus-v1.1.0-alpha.denovo.refined-from-transformer-greedy.csv",
    )
    v113_greedy = out_path(
        source,
        f"{base}.instanovo-1.1.3.transformer.model-instanovo-v1.1.0.denovo.greedy.beams-1.columns-preds.csv",
    )
    v113_beams = out_path(
        source,
        f"{base}.instanovo-1.1.3.transformer.model-instanovo-v1.1.0.denovo.beam-search.beams-5.save-beams.columns-preds.csv",
    )
    v113_plus = out_path(
        source,
        f"{base}.instanovo-1.1.3.instanovoplus.model-instanovoplus-v1.1.0.denovo.no-refinement.n-preds-5.csv",
    )
    v113_plus_verbose = out_path(
        source,
        f"{base}.instanovo-1.1.3.instanovoplus.model-instanovoplus-v1.1.0.denovo.no-refinement.n-preds-5.verbose-prediction-columns.csv",
    )
    v113_refined = out_path(
        source,
        f"{base}.instanovo-1.1.3.instanovoplus.model-instanovoplus-v1.1.0.denovo.refined-from-transformer-greedy.n-preds-5.csv",
    )
    v113_refined_verbose = out_path(
        source,
        f"{base}.instanovo-1.1.3.instanovoplus.model-instanovoplus-v1.1.0.denovo.refined-from-transformer-greedy.n-preds-5.verbose-prediction-columns.csv",
    )
    v122_greedy = out_path(
        source,
        f"{base}.instanovo-1.2.2.transformer.model-instanovo-v1.2.0.denovo.greedy.beams-1.normalized-columns.csv",
    )
    v122_beams = out_path(
        source,
        f"{base}.instanovo-1.2.2.transformer.model-instanovo-v1.2.0.denovo.beam-search.beams-5.save-all-predictions.normalized-columns.csv",
    )
    v122_plus = out_path(
        source,
        f"{base}.instanovo-1.2.2.instanovoplus.model-instanovoplus-v1.1.0.denovo.no-refinement.normalized-columns.csv",
    )
    v122_refined = out_path(
        source,
        f"{base}.instanovo-1.2.2.combined.model-instanovo-v1.2.0.instanovoplus-v1.1.0.denovo.refined.save-all-predictions.csv",
    )

    common = [
        f"batch_size={BATCH_SIZE}",
        f"num_workers={NUM_WORKERS}",
        "denovo=True",
        "subset=1.0",
        "fp16=True",
        "log_interval=1",
    ]
    old_device = ["device=cuda"]
    auto_device = ["device=auto"]

    jobs = [
        PredictionJob(
            "1.0.0-transformer-greedy",
            "1.0.0",
            [
                str(PYTHON),
                str(INOVO_ROOT / "1.0.0" / "instanovo" / "transformer" / "predict.py"),
                f"data_path={source.path}",
                "model_path=/workspace/models/instanovo-v1.0.0.ckpt",
                f"output_path={v100_greedy}",
                "num_beams=1",
                "save_beams=False",
                *common,
                *old_device,
            ],
            v100_greedy,
            "v1.0-transformer-greedy",
        ),
        PredictionJob(
            "1.0.0-transformer-beams5",
            "1.0.0",
            [
                str(PYTHON),
                str(INOVO_ROOT / "1.0.0" / "instanovo" / "transformer" / "predict.py"),
                f"data_path={source.path}",
                "model_path=/workspace/models/instanovo-v1.0.0.ckpt",
                f"output_path={v100_beams}",
                "num_beams=5",
                "save_beams=True",
                *common,
                *old_device,
            ],
            v100_beams,
            "v1.0-transformer-beams",
        ),
        PredictionJob(
            "1.1.0-transformer-greedy",
            "1.1.0",
            [
                str(INSTANOVO),
                "transformer",
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v110_greedy),
                "-i",
                "instanovo-v1.1.0",
                "--denovo",
                "num_beams=1",
                "save_beams=False",
                *common,
                *old_device,
            ],
            v110_greedy,
            "v1.1.0-transformer-greedy-predictions",
        ),
        PredictionJob(
            "1.1.0-transformer-beams5",
            "1.1.0",
            [
                str(INSTANOVO),
                "transformer",
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v110_beams),
                "-i",
                "instanovo-v1.1.0",
                "--denovo",
                "num_beams=5",
                "save_beams=True",
                *common,
                *old_device,
            ],
            v110_beams,
            "v1.1.0-transformer-beams-predictions",
        ),
        PredictionJob(
            "1.1.0-plus-standalone",
            "1.1.0",
            [
                str(INSTANOVO),
                "diffusion",
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v110_plus),
                "-p",
                "instanovoplus-v1.1.0-alpha",
                "--denovo",
                "--no-refinement",
                "-cn",
                "instanovoplus",
                "refine=False",
                *common,
                *old_device,
            ],
            v110_plus,
            "v1.1.0-instanovoplus-standalone",
        ),
        PredictionJob(
            "1.1.0-plus-refined",
            "1.1.0",
            [
                str(INSTANOVO),
                "diffusion",
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v110_refined),
                "-p",
                "instanovoplus-v1.1.0-alpha",
                "--denovo",
                "--with-refinement",
                "-cn",
                "default",
                f"instanovo_predictions_path={v110_greedy}",
                "id_col=spectrum_id",
                "pred_col=predictions",
                "+pred_tok_col=predictions_tokenised",
                "+log_probs_col=log_probabilities",
                "+token_log_probs_col=token_log_probabilities",
                *common,
                *old_device,
            ],
            v110_refined,
            "v1.1.0-instanovoplus-refined",
            depends_on=v110_greedy,
        ),
        PredictionJob(
            "1.1.3-transformer-greedy",
            "1.1.3",
            [
                str(INSTANOVO),
                "transformer",
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v113_greedy),
                "-i",
                "instanovo-v1.1.0",
                "--denovo",
                "num_beams=1",
                "save_beams=False",
                *common,
                *auto_device,
            ],
            v113_greedy,
            "v1.1.3-transformer-greedy-preds",
        ),
        PredictionJob(
            "1.1.3-transformer-beams5",
            "1.1.3",
            [
                str(INSTANOVO),
                "transformer",
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v113_beams),
                "-i",
                "instanovo-v1.1.0",
                "--denovo",
                "num_beams=5",
                "save_beams=True",
                *common,
                *auto_device,
            ],
            v113_beams,
            "v1.1.3-transformer-beams-preds",
        ),
        PredictionJob(
            "1.1.3-plus-standalone",
            "1.1.3",
            [
                str(INSTANOVO),
                "diffusion",
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v113_plus),
                "-p",
                "instanovoplus-v1.1.0",
                "--denovo",
                "--no-refinement",
                "-cn",
                "instanovoplus",
                "n_preds=5",
                "refine=False",
                "use_basic_logging=True",
                *common,
                *auto_device,
            ],
            v113_plus,
            "v1.1.3-instanovoplus-standalone",
        ),
        PredictionJob(
            "1.1.3-plus-standalone-verbose",
            "1.1.3",
            [
                str(INSTANOVO),
                "diffusion",
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v113_plus_verbose),
                "-p",
                "instanovoplus-v1.1.0",
                "--denovo",
                "--no-refinement",
                "-cn",
                "instanovoplus",
                "n_preds=5",
                "refine=False",
                "use_basic_logging=False",
                *common,
                *auto_device,
            ],
            v113_plus_verbose,
            "v1.1.3-instanovoplus-standalone-verbose",
        ),
        PredictionJob(
            "1.1.3-plus-refined",
            "1.1.3",
            [
                str(INSTANOVO),
                "diffusion",
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v113_refined),
                "-p",
                "instanovoplus-v1.1.0",
                "--denovo",
                "--with-refinement",
                "-cn",
                "instanovoplus",
                f"instanovo_predictions_path={v113_greedy}",
                "n_preds=5",
                "use_basic_logging=True",
                *common,
                *auto_device,
            ],
            v113_refined,
            "v1.1.3-instanovoplus-refined",
            depends_on=v113_greedy,
        ),
        PredictionJob(
            "1.1.3-plus-refined-verbose",
            "1.1.3",
            [
                str(INSTANOVO),
                "diffusion",
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v113_refined_verbose),
                "-p",
                "instanovoplus-v1.1.0",
                "--denovo",
                "--with-refinement",
                "-cn",
                "instanovoplus",
                f"instanovo_predictions_path={v113_greedy}",
                "n_preds=5",
                "use_basic_logging=False",
                *common,
                *auto_device,
            ],
            v113_refined_verbose,
            "v1.1.3-instanovoplus-refined-verbose",
            depends_on=v113_greedy,
        ),
        PredictionJob(
            "1.2.2-transformer-greedy",
            "1.2.2",
            [
                str(INSTANOVO),
                "transformer",
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v122_greedy),
                "-i",
                "instanovo-v1.2.0",
                "--denovo",
                "num_beams=1",
                "save_all_predictions=False",
                *common,
            ],
            v122_greedy,
            "v1.2.2-transformer-greedy-normalized",
        ),
        PredictionJob(
            "1.2.2-transformer-beams5",
            "1.2.2",
            [
                str(INSTANOVO),
                "transformer",
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v122_beams),
                "-i",
                "instanovo-v1.2.0",
                "--denovo",
                "num_beams=5",
                "save_all_predictions=True",
                *common,
            ],
            v122_beams,
            "v1.2.2-transformer-beams-normalized",
        ),
        PredictionJob(
            "1.2.2-plus-standalone",
            "1.2.2",
            [
                str(INSTANOVO),
                "diffusion",
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v122_plus),
                "-p",
                "instanovoplus-v1.1.0",
                "--denovo",
                "--no-refinement",
                "-cn",
                "instanovoplus",
                "refine=False",
                *common,
            ],
            v122_plus,
            "v1.2.2-instanovoplus-standalone-normalized",
        ),
        PredictionJob(
            "1.2.2-combined-refined",
            "1.2.2",
            [
                str(INSTANOVO),
                "predict",
                "-d",
                str(source.path),
                "-o",
                str(v122_refined),
                "-i",
                "instanovo-v1.2.0",
                "-p",
                "instanovoplus-v1.1.0",
                "--denovo",
                "--with-refinement",
                "num_beams=5",
                "save_all_predictions=True",
                *common,
            ],
            v122_refined,
            "v1.2.2-combined-refined-save-all",
        ),
    ]
    return jobs


def download_v100_model() -> None:
    model = MODEL_DIR / "instanovo-v1.0.0.ckpt"
    if model.exists():
        return
    url = "https://github.com/instadeepai/InstaNovo/releases/download/1.0.0/instanovo_extended.ckpt"
    log(f"Downloading InstaNovo v1.0.0 checkpoint -> {model}")
    urlretrieve(url, model)


def run_job(job: PredictionJob, source: SourceFile) -> dict[str, object]:
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    if download_remote_file(job.output_path):
        log(f"Skipping existing S3 prediction {job.output_path.name}")
        return inspect_output(job, source, "skipped_existing_s3", 0.0, None)

    if job.output_path.exists() and job.output_path.stat().st_size > 0:
        log(f"Skipping existing {job.output_path}")
        upload_file(job.output_path)
        return inspect_output(job, source, "skipped", 0.0, None)

    if job.depends_on is not None and not job.depends_on.exists():
        download_remote_file(job.depends_on)

    if job.depends_on is not None and not job.depends_on.exists():
        message = f"Missing dependency {job.depends_on}"
        log(f"Skipping {job.job_id}: {message}")
        return inspect_output(job, source, "skipped_missing_dependency", 0.0, message)

    log_name = f"{source.label}.{source.source_format.lower()}.{job.job_id}.log"
    log_path = LOG_DIR / log_name
    log(f"Running {job.job_id} on {source.source_format} -> {job.output_path.name}")
    log(f"Command: {' '.join(job.command)}")

    start = time.time()
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(
            job.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=run_env(job.version),
            cwd=ROOT,
        )
        assert proc.stdout is not None
        for chunk in iter(lambda: proc.stdout.readline(), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log_file.write(chunk)
            log_file.flush()
        return_code = proc.wait()

    elapsed = time.time() - start
    upload_file(log_path)
    if return_code != 0:
        message = f"Command failed with exit code {return_code}; see {log_path}"
        log(message)
        if not CONTINUE_ON_ERROR:
            raise RuntimeError(message)
        return inspect_output(job, source, "failed", elapsed, message)

    if not job.output_path.exists() or job.output_path.stat().st_size == 0:
        message = f"Command finished successfully but did not create a non-empty output: {job.output_path}"
        log(message)
        if not CONTINUE_ON_ERROR:
            raise RuntimeError(message)
        return inspect_output(job, source, "failed_missing_output", elapsed, message)

    upload_file(job.output_path)
    return inspect_output(job, source, "succeeded", elapsed, None)


def inspect_output(
    job: PredictionJob,
    source: SourceFile,
    status: str,
    elapsed_seconds: float,
    error: str | None,
) -> dict[str, object]:
    columns: list[str] = []
    rows = 0
    if job.output_path.exists() and job.output_path.stat().st_size > 0:
        with job.output_path.open(newline="") as handle:
            reader = csv.reader(handle)
            try:
                columns = next(reader)
            except StopIteration:
                columns = []
            rows = sum(1 for _ in reader)
    return {
        "status": status,
        "source_format": source.source_format,
        "source_label": source.label,
        "version": job.version,
        "job_id": job.job_id,
        "schema_group": job.schema_group,
        "output_path": str(job.output_path),
        "output_file": job.output_path.name,
        "rows": rows,
        "columns": columns,
        "column_count": len(columns),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "error": error,
    }


def upload_outputs() -> None:
    destination = upload_destination_root()
    if not destination:
        log("AICHOR_OUTPUT_PATH is not set; leaving outputs on local filesystem")
        return

    log(f"Uploading outputs to {destination}")
    for local_root in (OUT_DIR, LOG_DIR):
        if not local_root.exists():
            continue
        for path in local_root.rglob("*"):
            if not path.is_file():
                continue
            upload_file(path)


def write_manifest(results: list[dict[str, object]]) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": "smoke-first10" if MAX_SPECTRA > 0 else "full",
        "max_spectra": MAX_SPECTRA,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "mgf_url": MGF_URL,
        "mzml_url": MZML_URL,
        "results": results,
    }
    manifest_json = OUT_DIR / "prediction_manifest.json"
    manifest_csv = OUT_DIR / "prediction_manifest.csv"
    manifest_json.write_text(json.dumps(manifest, indent=2))
    with manifest_csv.open("w", newline="") as handle:
        fieldnames = [
            "status",
            "source_format",
            "source_label",
            "version",
            "job_id",
            "schema_group",
            "output_file",
            "rows",
            "column_count",
            "elapsed_seconds",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in fieldnames})
    return [manifest_json, manifest_csv]


def main() -> int:
    ensure_dirs()
    log(f"Running mode: {'smoke-first10' if MAX_SPECTRA > 0 else 'full'}")
    log(f"Using Python: {PYTHON}")
    log(f"Using InstaNovo CLI: {INSTANOVO}")
    subprocess.run([str(PYTHON), "-c", "import torch; print('cuda_available', torch.cuda.is_available(), 'device_count', torch.cuda.device_count())"], check=True)

    download_v100_model()
    sources = prepare_inputs()

    results: list[dict[str, object]] = []
    for source in sources:
        for job in build_jobs(source):
            results.append(run_job(job, source))
            for manifest_path in write_manifest(results):
                upload_file(manifest_path)

    failures = [r for r in results if r["status"] not in {"succeeded", "skipped", "skipped_existing_s3"}]
    for manifest_path in write_manifest(results):
        upload_file(manifest_path)
    upload_outputs()

    if failures:
        log(f"{len(failures)} job(s) failed")
        for failure in failures:
            log(f"FAILED {failure['source_format']} {failure['job_id']}: {failure['error']}")
        return 1

    log("All prediction jobs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
