#!/usr/bin/env python3
"""Setup for the Modly Wan2.2 TI2V 5B extension.

Modly calls this script as:
    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":"...","cuda_version":"..."}'

This setup intentionally does not clone Wan and does not download model weights.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXTENSION_ID = "wan22-ti2v-5b"
STATUS_SCHEMA = "modly.setup-status.v1"

RUNTIME_REQUIREMENTS = "runtime_requirements.txt"
VENDOR_RUNTIME = Path("vendor") / "wan-runtime"
VENDOR_SENTINEL = VENDOR_RUNTIME / "wan" / "textimage2video.py"
RUNTIME_ROOT = ".wan22-ti2v-runtime"
FLASH_ATTN_WHEELHOUSE_RELATIVE = Path("wheelhouse") / "flash-attn"

TORCH_INDEX_URL_CU128 = "https://download.pytorch.org/whl/cu128"
TORCH_PACKAGES_CU128 = ["torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0"]
PIP_FLAGS = ["--no-cache-dir", "--retries", "5", "--timeout", "60"]
FLASH_ATTN_PACKAGE = "flash-attn"
FLASH_ATTN_BUILD_REQUIREMENTS = ["psutil", "ninja"]
MIN_FLASH_ATTN_SM = 80

MANAGED_FLASH_ATTN_WHEELS = [
    {
        "id": "windows-cp311-cu128-torch2.7",
        "python_tag": "cp311",
        "abi_tag": "cp311",
        "platform_tag": "win_amd64",
        "filename": "flash_attn-2.8.3+cu128torch2.7-cp311-cp311-win_amd64.whl",
        "url": "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/flash_attn-latest/flash_attn-2.8.3%2Bcu128torch2.7-cp311-cp311-win_amd64.whl",
        "sha256": "4e935a37ac8ef2dd71836d65bf560507617c316e176a068e5e1f4ea5248e0cde",
        "size_bytes": 250850103,
    },
]


def log(message: str) -> None:
    print(f"[setup:{EXTENSION_ID}] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    args = list(sys.argv[1:] if argv is None else argv)
    options: dict[str, Any] = {
        "allow_flash_attn_source_build": os.environ.get("WAN22_ALLOW_FLASH_ATTN_SOURCE_BUILD") == "1",
        "build_flash_attn_wheel": False,
        "max_build_jobs": None,
    }
    payload_args: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--allow-flash-attn-source-build":
            options["allow_flash_attn_source_build"] = True
        elif arg == "--build-flash-attn-wheel":
            options["build_flash_attn_wheel"] = True
        elif arg == "--max-build-jobs":
            index += 1
            if index >= len(args):
                raise SystemExit("--max-build-jobs requires a value")
            options["max_build_jobs"] = args[index]
        elif arg.startswith("--max-build-jobs="):
            options["max_build_jobs"] = arg.split("=", 1)[1]
        else:
            payload_args.append(arg)
        index += 1

    if len(payload_args) == 1:
        try:
            payload = json.loads(payload_args[0])
        except json.JSONDecodeError as exc:
            raise SystemExit(f"setup payload must be valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit("setup payload must be a JSON object")
        return normalize_payload(payload), options

    if len(payload_args) >= 2:
        payload: dict[str, Any] = {"python_exe": payload_args[0], "ext_dir": payload_args[1]}
        if len(payload_args) >= 3:
            payload["gpu_sm"] = payload_args[2]
        return normalize_payload(payload), options

    raise SystemExit("Usage: python setup.py '<json-payload>' [--build-flash-attn-wheel] [--allow-flash-attn-source-build]")


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    python_exe = payload.get("python_exe")
    ext_dir = payload.get("ext_dir")
    if not isinstance(python_exe, str) or not python_exe.strip():
        raise SystemExit("setup payload must include a non-empty python_exe")
    if not isinstance(ext_dir, str) or not ext_dir.strip():
        raise SystemExit("setup payload must include a non-empty ext_dir")

    return {
        "python_exe": python_exe.strip(),
        "ext_dir": str(Path(ext_dir.strip()).expanduser().resolve()),
        "gpu_sm": str(payload.get("gpu_sm", payload.get("gpuSm", "")) or ""),
        "cuda_version": str(payload.get("cuda_version", payload.get("cudaVersion", "")) or ""),
    }


def venv_python(venv_dir: Path) -> Path:
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    log("$ " + " ".join(cmd))
    run_env = {**os.environ, **env} if env else None
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=run_env, text=True, check=check)


def write_status(ext_dir: Path, payload: dict[str, Any], status: str, error: str | None = None) -> None:
    status_dir = ext_dir / ".modly" / "setup"
    status_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "schema": STATUS_SCHEMA,
        "extension_id": EXTENSION_ID,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "payload": {
            "gpu_sm": payload.get("gpu_sm", ""),
            "cuda_version": payload.get("cuda_version", ""),
        },
        "policy": {
            "clones_upstream_repo": False,
            "downloads_model_weights": False,
            "weights_download_surface": "Modly Models UI",
            "flash_attn_required": True,
            "flash_attn_install_order": ["local_wheelhouse", "managed_wheel", "binary_only_pip", "explicit_source_build"],
            "torch_lane": {"index_url": TORCH_INDEX_URL_CU128, "packages": TORCH_PACKAGES_CU128},
        },
    }
    if error:
        data["error"] = error
    (status_dir / "setup-status.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def ensure_vendor_runtime(ext_dir: Path) -> None:
    sentinel = ext_dir / VENDOR_SENTINEL
    if not sentinel.exists():
        raise RuntimeError(
            f"Vendored Wan runtime is missing: {sentinel}. "
            "Repair the extension package; setup will not clone Wan."
        )
    log(f"Found vendored Wan runtime at {ext_dir / VENDOR_RUNTIME}")


def _normalize_gpu_sm(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    try:
        text = str(value).strip().lower().replace("sm_", "").replace("sm", "")
        if "." in text:
            major, minor = text.split(".", 1)
            return int(major) * 10 + int(minor[:1] or "0")
        number = int(text)
        return number * 10 if number < 10 else number
    except (TypeError, ValueError):
        return None


def preflight_gpu_requirement(payload: dict[str, Any]) -> None:
    sm = _normalize_gpu_sm(payload.get("gpu_sm"))
    if sm is None:
        log("GPU compute capability was not provided; runtime will verify sm_80+ before generation")
        return
    if sm < MIN_FLASH_ATTN_SM:
        raise RuntimeError(
            f"Wan2.2 TI2V requires FlashAttention and an Ampere/RTX 30-series or newer CUDA GPU. "
            f"Detected sm_{sm}; minimum is sm_{MIN_FLASH_ATTN_SM}."
        )
    log(f"GPU capability preflight passed: sm_{sm}")


def _parse_cuda_version(cuda_version: str) -> tuple[int, int] | None:
    """Parse Modly's compact form (128) and conventional dotted form (12.8).

    Empty input means that the host did not report a CUDA version. Malformed or
    ambiguous values are rejected instead of being coerced into a supported lane.
    """
    normalized = cuda_version.strip()
    if not normalized:
        return None

    dotted = re.fullmatch(r"(\\d{1,2})\\.(\\d)", normalized)
    if dotted:
        return int(dotted.group(1)), int(dotted.group(2))

    compact = re.fullmatch(r"(\\d{2})(\\d)", normalized)
    if compact:
        return int(compact.group(1)), int(compact.group(2))

    major_only = re.fullmatch(r"\\d{2}", normalized)
    if major_only:
        return int(normalized), 0

    raise ValueError(f"unsupported CUDA version format: {cuda_version!r}")


def torch_lane_supported(cuda_version: str) -> bool:
    try:
        parsed = _parse_cuda_version(cuda_version)
    except ValueError:
        return False
    return parsed is None or parsed >= (12, 8)


def create_venv(payload: dict[str, Any], ext_dir: Path) -> Path:
    venv_dir = ext_dir / "venv"
    if venv_python(venv_dir).exists():
        log(f"Reusing existing venv at {venv_dir}")
        return venv_dir

    log(f"Creating venv at {venv_dir}")
    run([payload["python_exe"], "-m", "venv", str(venv_dir)])
    return venv_dir


def cleanup_unsupported_cuda_residue(py: Path) -> None:
    # Torch 2.7.0+cu128 on this Linux ARM64 lane does not require the split
    # NVIDIA cuSPARSELt package. It can be left behind when downgrading from
    # newer Torch wheels and makes `pip check` fail as unsupported on platform.
    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64"}:
        return
    log("Removing unsupported CUDA package residue if present: nvidia-cusparselt-cu12")
    run([str(py), "-m", "pip", "uninstall", "-y", "nvidia-cusparselt-cu12"], check=False)


def runtime_root(ext_dir: Path) -> Path:
    return ext_dir / RUNTIME_ROOT


def flash_attn_wheelhouse(ext_dir: Path) -> Path:
    return runtime_root(ext_dir) / FLASH_ATTN_WHEELHOUSE_RELATIVE


def local_flash_attn_wheels(ext_dir: Path) -> list[Path]:
    wheelhouse = flash_attn_wheelhouse(ext_dir)
    if not wheelhouse.exists():
        return []
    return sorted([*wheelhouse.glob("flash_attn-*.whl"), *wheelhouse.glob("flash-attn-*.whl")])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_python_tags() -> dict[str, str]:
    implementation = getattr(sys.implementation, "name", "")
    python_tag = f"py{sys.version_info.major}" if implementation != "cpython" else f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_tag = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return {"python_tag": python_tag, "abi_tag": python_tag, "platform_tag": platform_tag}


def venv_python_tags(py: Path) -> dict[str, str]:
    code = """
import json, sys, sysconfig
implementation = getattr(sys.implementation, 'name', '')
python_tag = f'py{sys.version_info.major}' if implementation != 'cpython' else f'cp{sys.version_info.major}{sys.version_info.minor}'
platform_tag = sysconfig.get_platform().replace('-', '_').replace('.', '_')
print(json.dumps({'python_tag': python_tag, 'abi_tag': python_tag, 'platform_tag': platform_tag}, sort_keys=True))
"""
    result = subprocess.run([str(py), "-c", code], text=True, capture_output=True, check=False)
    if result.returncode == 0:
        try:
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            pass
    return current_python_tags()


def managed_flash_attn_wheel_for_tags(tags: dict[str, str]) -> dict[str, Any] | None:
    for wheel in MANAGED_FLASH_ATTN_WHEELS:
        if all(tags.get(key) == wheel[key] for key in ["python_tag", "abi_tag", "platform_tag"]):
            return wheel
    return None


def download_managed_flash_attn_wheel(ext_dir: Path, py: Path) -> dict[str, Any] | None:
    tags = venv_python_tags(py)
    wheel = managed_flash_attn_wheel_for_tags(tags)
    if wheel is None:
        log(
            "No managed flash-attn wheel is registered for "
            f"{tags.get('python_tag')}-{tags.get('abi_tag')}-{tags.get('platform_tag')}"
        )
        return None

    wheelhouse = flash_attn_wheelhouse(ext_dir)
    wheelhouse.mkdir(parents=True, exist_ok=True)
    destination = wheelhouse / str(wheel["filename"])
    if not destination.exists():
        log(f"Downloading managed flash-attn wheel: {wheel['id']}")
        urllib.request.urlretrieve(str(wheel["url"]), destination)
    else:
        log(f"Using cached managed flash-attn wheel: {destination.name}")

    actual_size = destination.stat().st_size
    actual_sha256 = sha256_file(destination)
    if actual_size != int(wheel["size_bytes"]) or actual_sha256 != wheel["sha256"]:
        try:
            destination.unlink()
        except OSError:
            pass
        raise RuntimeError(
            "Managed flash-attn wheel checksum verification failed: "
            f"expected {wheel['sha256']} / {wheel['size_bytes']} bytes, got {actual_sha256} / {actual_size} bytes"
        )
    return {"status": "downloaded", "wheel": wheel, "path": str(destination), "tags": tags}


def install_flash_attn_from_wheelhouse(ext_dir: Path, py: Path, *, mode: str) -> None:
    wheelhouse = flash_attn_wheelhouse(ext_dir)
    wheels = local_flash_attn_wheels(ext_dir)
    if not wheels:
        raise RuntimeError(f"No flash-attn wheels found in {wheelhouse}")
    log(f"Installing flash-attn from {mode} wheelhouse: {wheelhouse}")
    log("Available flash-attn wheels: " + ", ".join(path.name for path in wheels))
    run([
        str(py),
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        "--no-index",
        "--find-links",
        str(wheelhouse),
        FLASH_ATTN_PACKAGE,
    ])


def cuda_build_env(torch_cuda_version: str = "12.8") -> dict[str, str]:
    candidates = [
        Path(f"/usr/local/cuda-{torch_cuda_version}"),
        Path(f"/usr/local/cuda-{torch_cuda_version.split('.', 1)[0]}"),
    ]
    for cuda_home in candidates:
        if (cuda_home / "bin" / "nvcc").exists():
            return {
                "CUDA_HOME": str(cuda_home),
                "CUDA_PATH": str(cuda_home),
                "PATH": str(cuda_home / "bin") + os.pathsep + os.environ.get("PATH", ""),
                "LD_LIBRARY_PATH": str(cuda_home / "lib64") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
            }
    return {}


def install_flash_attn(ext_dir: Path, py: Path, *, allow_source_build: bool) -> None:
    flash_attn_wheelhouse(ext_dir).mkdir(parents=True, exist_ok=True)
    if local_flash_attn_wheels(ext_dir):
        install_flash_attn_from_wheelhouse(ext_dir, py, mode="local")
        return

    if download_managed_flash_attn_wheel(ext_dir, py) is not None:
        install_flash_attn_from_wheelhouse(ext_dir, py, mode="managed")
        return

    log("No local flash-attn wheel found; trying binary-only pip install before any source build")
    binary = run([str(py), "-m", "pip", "install", *PIP_FLAGS, "--only-binary", ":all:", FLASH_ATTN_PACKAGE], check=False)
    if binary.returncode == 0:
        return

    if not allow_source_build:
        raise RuntimeError(
            "flash-attn-wheel-unavailable: no compatible flash-attn wheel was available. "
            f"Put a wheel in {flash_attn_wheelhouse(ext_dir)} or run setup.py --build-flash-attn-wheel with the same Modly payload, "
            "then rerun normal setup. Source build is disabled by default because it is slow and platform-sensitive."
        )

    env = cuda_build_env("12.8")
    if env.get("CUDA_HOME"):
        log(f"Using CUDA_HOME={env['CUDA_HOME']} for explicit flash-attn source build")
    log("Explicit source build is enabled; this can take a long time")
    run([str(py), "-m", "pip", "install", *PIP_FLAGS, FLASH_ATTN_PACKAGE, "--no-build-isolation"], env=env or None)


def install_dependencies(ext_dir: Path, venv_dir: Path, payload: dict[str, Any], options: dict[str, Any]) -> None:
    py = str(venv_python(venv_dir))
    run([py, "-m", "pip", "install", *PIP_FLAGS, "--upgrade", "pip", "setuptools", "wheel"])

    if not torch_lane_supported(payload.get("cuda_version", "")):
        raise RuntimeError(
            "Wan2.2 TI2V setup currently supports the verified cu128 PyTorch/flash-attn lane only. "
            f"Detected cuda_version={payload.get('cuda_version', '')!r}."
        )

    if os.environ.get("WAN22_SKIP_TORCH_INSTALL") == "1":
        log("WAN22_SKIP_TORCH_INSTALL=1; skipping torch wheel installation")
    else:
        log(f"Installing exact PyTorch CUDA lane from {TORCH_INDEX_URL_CU128}: {', '.join(TORCH_PACKAGES_CU128)}")
        run([py, "-m", "pip", "install", *PIP_FLAGS, *TORCH_PACKAGES_CU128, "--index-url", TORCH_INDEX_URL_CU128])

    cleanup_unsupported_cuda_residue(Path(py))

    requirements = ext_dir / RUNTIME_REQUIREMENTS
    if not requirements.exists():
        raise RuntimeError(f"Missing {requirements}")
    log("Installing Wan runtime dependencies")
    run([py, "-m", "pip", "install", *PIP_FLAGS, "-r", str(requirements)])

    log("Installing flash-attn build prerequisites")
    run([py, "-m", "pip", "install", *PIP_FLAGS, *FLASH_ATTN_BUILD_REQUIREMENTS])
    install_flash_attn(ext_dir, Path(py), allow_source_build=bool(options.get("allow_flash_attn_source_build")))


def run_import_probes(ext_dir: Path, venv_dir: Path) -> None:
    py = str(venv_python(venv_dir))
    code = (
        "import json, sys; "
        "from pathlib import Path; "
        f"sys.path.insert(0, {str(ext_dir / VENDOR_RUNTIME)!r}); "
        "import torch, flash_attn; "
        "from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE; "
        "assert torch.cuda.is_available(), 'torch CUDA is not available'; "
        "assert FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE, 'Wan flash_attn backend is unavailable'; "
        "from wan.textimage2video import WanTI2V; "
        "from wan.configs.wan_ti2v_5B import ti2v_5B; "
        "props = torch.cuda.get_device_properties(0); "
        "print(json.dumps({'status':'wan_ti2v_runtime_ok','frame_num':ti2v_5B.frame_num,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'flash_attn':getattr(flash_attn,'__version__',None),'device':props.name,'sm':f'{props.major}.{props.minor}'}, sort_keys=True))"
    )
    run([py, "-c", code])


def build_flash_attn_wheel(payload: dict[str, Any], options: dict[str, Any]) -> None:
    ext_dir = Path(payload["ext_dir"])
    ensure_vendor_runtime(ext_dir)
    preflight_gpu_requirement(payload)
    venv_dir = create_venv(payload, ext_dir)
    py = venv_python(venv_dir)
    wheelhouse = flash_attn_wheelhouse(ext_dir)
    wheelhouse.mkdir(parents=True, exist_ok=True)

    log("Preparing venv for flash-attn wheel build")
    run([str(py), "-m", "pip", "install", *PIP_FLAGS, "--upgrade", "pip", "setuptools", "wheel", *FLASH_ATTN_BUILD_REQUIREMENTS])
    log(f"Installing exact PyTorch CUDA lane for wheel build from {TORCH_INDEX_URL_CU128}")
    run([str(py), "-m", "pip", "install", *PIP_FLAGS, *TORCH_PACKAGES_CU128, "--index-url", TORCH_INDEX_URL_CU128])

    env = cuda_build_env("12.8")
    max_jobs = options.get("max_build_jobs")
    if max_jobs:
        env = {**env, "MAX_JOBS": str(max_jobs)}
        log(f"Using MAX_JOBS={max_jobs} for flash-attn wheel build")
    if env.get("CUDA_HOME"):
        log(f"Using CUDA_HOME={env['CUDA_HOME']} for flash-attn wheel build")
    log(f"Building flash-attn wheel into {wheelhouse}")
    run([
        str(py),
        "-m",
        "pip",
        "wheel",
        *PIP_FLAGS,
        FLASH_ATTN_PACKAGE,
        "--no-build-isolation",
        "--no-deps",
        "--wheel-dir",
        str(wheelhouse),
    ], env=env or None)
    wheels = local_flash_attn_wheels(ext_dir)
    if not wheels:
        raise RuntimeError(f"flash-attn wheel build completed but no wheel was found in {wheelhouse}")
    log("flash-attn wheel ready: " + ", ".join(path.name for path in wheels))


def setup(payload: dict[str, Any], options: dict[str, Any]) -> None:
    ext_dir = Path(payload["ext_dir"])
    write_status(ext_dir, payload, "running")
    ensure_vendor_runtime(ext_dir)
    preflight_gpu_requirement(payload)
    venv_dir = create_venv(payload, ext_dir)
    install_dependencies(ext_dir, venv_dir, payload, options)
    run_import_probes(ext_dir, venv_dir)
    write_status(ext_dir, payload, "ready")
    log("Setup complete. Download Wan-AI/Wan2.2-TI2V-5B weights from the Modly Models UI before generation.")


def main(argv: list[str] | None = None) -> int:
    payload, options = parse_args(argv)
    ext_dir = Path(payload["ext_dir"])
    try:
        if options.get("build_flash_attn_wheel"):
            build_flash_attn_wheel(payload, options)
            return 0
        setup(payload, options)
    except Exception as exc:
        try:
            write_status(ext_dir, payload, "error", str(exc))
        finally:
            log(f"ERROR: {exc}")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
