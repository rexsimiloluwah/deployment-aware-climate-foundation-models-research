from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import modal


APP_NAME = "geoai-aurora-adaptation"
VOLUME_NAME = "geoai-aurora-artifacts"
SECRET_NAME = "cds-api"
GPU_TYPE = "A100-80GB"
REMOTE_ROOT = Path("/workspace")
VOLUME_ROOT = Path("/mnt/geoai")
STATIC_PATH = VOLUME_ROOT / "data" / "weatherbench2" / "hres_t0_global" / "static.nc"
LOCAL_ROOT = Path(__file__).resolve().parents[1]


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "cdsapi>=0.7.5",
        "dask>=2024.8.0",
        "gcsfs>=2024.6.0",
        "huggingface-hub>=0.26.0",
        "matplotlib>=3.8.0",
        "microsoft-aurora>=2.0.0",
        "netcdf4>=1.7.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "pyyaml>=6.0.0",
        "torch>=2.4.0",
        "xarray>=2024.7.0",
        "zarr>=2.18.0",
    )
    .add_local_dir(LOCAL_ROOT / "scripts", remote_path=str(REMOTE_ROOT / "scripts"))
    .add_local_dir(LOCAL_ROOT / "eval_harness", remote_path=str(REMOTE_ROOT / "eval_harness"))
    .add_local_dir(LOCAL_ROOT / "config", remote_path=str(REMOTE_ROOT / "config"))
    .add_local_file(LOCAL_ROOT / "pyproject.toml", remote_path=str(REMOTE_ROOT / "pyproject.toml"))
    .add_local_file(LOCAL_ROOT / "README.md", remote_path=str(REMOTE_ROOT / "README.md"))
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image, secrets=[modal.Secret.from_name(SECRET_NAME)], volumes={VOLUME_ROOT: volume})


def run_command(args: list[str], env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    merged_env["PYTHONPATH"] = str(REMOTE_ROOT)
    merged_env["AURORA_STATIC_FIELDS_URI"] = str(STATIC_PATH)
    merged_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if env:
        merged_env.update(env)
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=REMOTE_ROOT, env=merged_env, check=True)


def ensure_static_fields() -> None:
    if STATIC_PATH.exists():
        print(f"Static fields already exist in Modal volume: {STATIC_PATH}", flush=True)
        return
    STATIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "python",
            "scripts/download_era5_static_fields.py",
            "--global-domain",
            "--output",
            str(STATIC_PATH),
        ]
    )


def copy_artifacts_to_volume(run_label: str) -> None:
    source = REMOTE_ROOT / "artifacts" / run_label
    destination = VOLUME_ROOT / "artifacts" / run_label
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    volume.commit()
    print(f"Copied artifacts to Modal volume: {destination}", flush=True)


def run_aurora_adaptation(run_label: str, max_times: int, train_steps: int, lead_times: str) -> None:
    run_command(
        [
            "python",
            "scripts/run_aurora_forecasting_experiment.py",
            "--model-adapters",
            "aurora_small:linear_calibration,aurora_small:lora",
            "--split",
            "val",
            "--train-split",
            "train",
            "--lead-times-hours",
            lead_times,
            "--max-initialization-times",
            str(max_times),
            "--lora-train-steps",
            str(train_steps),
            "--initialization-sampling",
            "even",
            "--device",
            "cuda",
            "--precision",
            "bf16",
            "--run-label",
            run_label,
        ]
    )
    run_command(
        [
            "python",
            "scripts/check_aurora_results_schema.py",
            "--results",
            f"artifacts/{run_label}/results.jsonl",
            "--require-adapters",
            "linear_calibration,lora",
        ]
    )
    run_command(["python", "scripts/analyze_aurora_results.py", "--artifact-dir", f"artifacts/{run_label}"])
    copy_artifacts_to_volume(run_label)


@app.function(gpu=GPU_TYPE, timeout=60 * 60 * 8, volumes={VOLUME_ROOT: volume})
def run_canary() -> str:
    ensure_static_fields()
    run_aurora_adaptation(
        run_label="aurora_greater_horn_modal_canary_v1",
        max_times=2,
        train_steps=2,
        lead_times="6,12",
    )
    return "aurora_greater_horn_modal_canary_v1"


@app.function(gpu=GPU_TYPE, timeout=60 * 60 * 24, volumes={VOLUME_ROOT: volume})
def run_full() -> str:
    ensure_static_fields()
    run_aurora_adaptation(
        run_label="aurora_greater_horn_adapted_v1",
        max_times=64,
        train_steps=64,
        lead_times="6,12,24,48",
    )
    return "aurora_greater_horn_adapted_v1"


@app.local_entrypoint()
def main(mode: str = "canary", detach: bool = False) -> None:
    # On a flaky link, .remote() (blocking) is cancelled when the client disconnects, even under
    # `modal run --detach`. For the long full run, spawn a truly detached call that runs to
    # completion server-side; poll the volume for artifacts afterward.
    fn = run_canary if mode == "canary" else run_full if mode == "full" else None
    if fn is None:
        raise ValueError("mode must be 'canary' or 'full'")
    if detach:
        call = fn.spawn()
        print(f"Spawned detached Modal {mode} run: function_call_id={call.object_id}")
        print("It runs to completion server-side. Poll: modal app list / the volume artifacts dir.")
        return
    label = fn.remote()
    print(f"Completed Modal Aurora run: {label}")
    print(f"Pull with: modal volume get {VOLUME_NAME} artifacts/{label} artifacts/{label} --force")
