from __future__ import annotations

import importlib
import io
import logging
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from services.generators.base import BaseGenerator, GenerationCancelled
except ModuleNotFoundError:  # pragma: no cover - standalone contract checks
    class GenerationCancelled(Exception):
        pass

    class BaseGenerator:  # type: ignore[override]
        MODEL_ID = ""
        DISPLAY_NAME = ""
        VRAM_GB = 0

        def __init__(self, model_dir: Path | None = None, outputs_dir: Path | None = None) -> None:
            self.model_dir = Path(model_dir or ".")
            self.outputs_dir = Path(outputs_dir or ".")
            self._model = None
            self.hf_repo = ""
            self.hf_skip_prefixes: list[str] = []
            self.download_check = ""
            self._params_schema: list[dict[str, Any]] = []

        def is_downloaded(self) -> bool:
            if self.download_check:
                return (self.model_dir / self.download_check).exists()
            return self.model_dir.exists() and any(self.model_dir.iterdir())

        def _check_cancelled(self, cancel_event: Optional[threading.Event]) -> None:
            if cancel_event and cancel_event.is_set():
                raise GenerationCancelled()


EXTENSION_DIR = Path(__file__).resolve().parent
WAN_RUNTIME_DIR = EXTENSION_DIR / "vendor" / "wan-runtime"
if str(WAN_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(WAN_RUNTIME_DIR))

LOGGER = logging.getLogger("modly.wan22-ti2v")
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="[wan22-ti2v] %(message)s")

MODEL_REPO = "Wan-AI/Wan2.2-TI2V-5B"
DOWNLOAD_SENTINEL = "Wan2.2_VAE.pth"
MIN_VRAM_GB = 24
MIN_FLASH_ATTN_SM = 80
FPS = 24
MAX_AREA_720P = 1280 * 704

PARAMS_SCHEMA: list[dict[str, Any]] = [
    {
        "id": "prompt",
        "label": "Prompt",
        "type": "string",
        "default": "A cinematic shot with subtle camera motion, detailed lighting, and smooth natural movement.",
        "tooltip": "Positive motion/video prompt. This is required for Wan TI2V generation.",
    },
    {
        "id": "negative_prompt",
        "label": "Negative Prompt",
        "type": "string",
        "default": "",
        "tooltip": "Optional negative prompt. Leave empty to use Wan's default negative prompt.",
    },
    {
        "id": "orientation",
        "label": "Orientation",
        "type": "select",
        "default": "auto",
        "options": [
            {"value": "auto", "label": "Auto"},
            {"value": "landscape_720p", "label": "Landscape 720p"},
            {"value": "portrait_720p", "label": "Portrait 720p"},
        ],
    },
    {
        "id": "duration",
        "label": "Frames",
        "type": "select",
        "default": "81",
        "options": [
            {"value": "81", "label": "81 frames (~3.4s)"},
            {"value": "121", "label": "121 frames (~5.0s)"},
        ],
    },
    {"id": "steps", "label": "Steps", "type": "int", "default": 20, "min": 20, "max": 50},
    {
        "id": "guidance_scale",
        "label": "Guidance Scale",
        "type": "float",
        "default": 5.0,
        "min": 1.0,
        "max": 10.0,
        "step": 0.1,
    },
    {"id": "seed", "label": "Seed", "type": "int", "default": -1, "min": -1, "max": 2147483647},
]


def _progress(progress_cb: Optional[Callable[[int, str], None]], pct: int, label: str) -> None:
    if progress_cb:
        progress_cb(pct, label)


def _as_int(value: Any, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _as_float(value: Any, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed


def _sanitize_filename(value: str, fallback: str = "wan22_ti2v") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-._")
    return (cleaned or fallback)[:80]


def _target_size_for_orientation(orientation: str, image_width: int, image_height: int) -> tuple[int, int]:
    if orientation == "landscape_720p":
        return 1280, 704
    if orientation == "portrait_720p":
        return 704, 1280
    return (1280, 704) if image_width >= image_height else (704, 1280)


def _cover_crop(image, target_width: int, target_height: int):
    target_aspect = target_width / target_height
    source_aspect = image.width / image.height

    if abs(source_aspect - target_aspect) < 0.01:
        return image

    if source_aspect > target_aspect:
        new_width = int(round(image.height * target_aspect))
        left = max(0, (image.width - new_width) // 2)
        return image.crop((left, 0, left + new_width, image.height))

    new_height = int(round(image.width / target_aspect))
    top = max(0, (image.height - new_height) // 2)
    return image.crop((0, top, image.width, top + new_height))


class Wan22TI2VGenerator(BaseGenerator):
    MODEL_ID = "wan22-ti2v-5b/image-to-video"
    DISPLAY_NAME = "Wan2.2 TI2V 5B"
    VRAM_GB = MIN_VRAM_GB

    @classmethod
    def params_schema(cls) -> list[dict[str, Any]]:
        return PARAMS_SCHEMA

    def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
        super().__init__(model_dir, outputs_dir)
        self.download_check = self.download_check or DOWNLOAD_SENTINEL
        self._device_id = 0

    def _weights_missing_message(self) -> str:
        return (
            f"Wan2.2 TI2V 5B weights are missing in {self.model_dir}. "
            f"Download {MODEL_REPO} from the Modly Models UI before generation. "
            "setup.py intentionally does not download model weights."
        )

    def _validate_runtime_files(self) -> None:
        sentinel = WAN_RUNTIME_DIR / "wan" / "textimage2video.py"
        if not sentinel.exists():
            raise RuntimeError(
                f"Vendored Wan runtime is missing: {sentinel}. Repair the extension; setup.py will not clone Wan."
            )

    def _flash_attn_status(self) -> tuple[bool, str, dict[str, Any]]:
        try:
            module = importlib.import_module("flash_attn")
        except Exception as exc:  # pragma: no cover - depends on local venv
            return (
                False,
                "flash-attn is not installed. Rerun extension setup; Wan2.2 TI2V requires FlashAttention and no SDPA fallback is enabled.",
                {"error": f"{type(exc).__name__}: {exc}"},
            )

        try:
            from wan.modules.attention import FLASH_ATTN_2_AVAILABLE, FLASH_ATTN_3_AVAILABLE
        except Exception as exc:  # pragma: no cover - depends on vendored runtime
            return False, f"Wan attention backend probe failed: {exc}", {"flash_attn_version": getattr(module, "__version__", None)}

        if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
            return (
                False,
                "flash-attn imports, but Wan did not detect a usable FlashAttention 2/3 backend.",
                {
                    "flash_attn_version": getattr(module, "__version__", None),
                    "flash_attn_2_available": FLASH_ATTN_2_AVAILABLE,
                    "flash_attn_3_available": FLASH_ATTN_3_AVAILABLE,
                },
            )

        return (
            True,
            "FlashAttention ready.",
            {
                "flash_attn_version": getattr(module, "__version__", None),
                "flash_attn_2_available": FLASH_ATTN_2_AVAILABLE,
                "flash_attn_3_available": FLASH_ATTN_3_AVAILABLE,
            },
        )

    def _torch_status(self) -> tuple[bool, str, dict[str, Any]]:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on local venv
            return False, f"PyTorch import failed: {exc}", {}

        details: dict[str, Any] = {"torch_version": getattr(torch, "__version__", None), "torch_cuda_version": getattr(torch.version, "cuda", None)}
        if not torch.cuda.is_available():
            return False, "CUDA is not available. Wan2.2 TI2V requires an NVIDIA CUDA GPU.", details

        props = torch.cuda.get_device_properties(self._device_id)
        capability = torch.cuda.get_device_capability(self._device_id)
        sm = capability[0] * 10 + capability[1]
        vram_gb = props.total_memory / (1024 ** 3)
        details.update({"device": props.name, "detected_vram_gb": vram_gb, "compute_capability": f"sm_{sm}", "minimum_sm": MIN_FLASH_ATTN_SM, "minimum_vram_gb": MIN_VRAM_GB})

        if sm < MIN_FLASH_ATTN_SM:
            return False, f"Detected compute capability sm_{sm}; Wan2.2 TI2V requires sm_{MIN_FLASH_ATTN_SM}+ for FlashAttention.", details

        flash_ok, flash_reason, flash_details = self._flash_attn_status()
        details.update(flash_details)
        if not flash_ok:
            return False, flash_reason, details

        if vram_gb < MIN_VRAM_GB:
            return False, f"Detected {vram_gb:.1f} GB VRAM; Wan2.2 TI2V 5B requires about {MIN_VRAM_GB} GB with offload.", details

        return True, f"CUDA and FlashAttention ready on {props.name} with {vram_gb:.1f} GB VRAM.", details

    def readiness_status(self) -> dict[str, Any]:
        try:
            self._validate_runtime_files()
        except Exception as exc:
            return {
                "ok": False,
                "machine_code": "runtime_missing",
                "label_hint": "Runtime missing",
                "reason": str(exc),
            }

        cuda_ok, reason, details = self._torch_status()
        if not cuda_ok:
            return {
                "ok": False,
                "machine_code": "runtime_not_ready",
                "label_hint": "Setup incomplete",
                "reason": reason,
                "details": details,
            }

        if not self.is_downloaded():
            # Model weights are managed by Modly's Models UI. Runtime readiness
            # must not surface a label here, otherwise ExtensionCard prioritizes
            # the readiness label over the standard Download button.
            return {
                "ok": True,
                "machine_code": "runtime_ready_weights_pending",
                "reason": self._weights_missing_message(),
                "details": {**details, "hf_repo": MODEL_REPO, "download_check": self.download_check or DOWNLOAD_SENTINEL},
            }

        return {
            "ok": True,
            "machine_code": "ready",
            "label_hint": "Ready",
            "reason": reason,
            "details": details,
        }

    def is_downloaded(self) -> bool:
        return (self.model_dir / (self.download_check or DOWNLOAD_SENTINEL)).exists()

    def load(self) -> None:
        if self._model is not None:
            return

        self._validate_runtime_files()
        cuda_ok, reason, _ = self._torch_status()
        if not cuda_ok:
            raise RuntimeError(reason)
        if not self.is_downloaded():
            raise RuntimeError(self._weights_missing_message())

        from wan.configs.wan_ti2v_5B import ti2v_5B
        from wan.textimage2video import WanTI2V

        LOGGER.info("Loading Wan2.2 TI2V 5B runtime from %s", self.model_dir)
        self._model = WanTI2V(
            config=ti2v_5B,
            checkpoint_dir=str(self.model_dir),
            device_id=self._device_id,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=True,
            init_on_cpu=True,
            convert_model_dtype=True,
        )
        LOGGER.info("Wan2.2 TI2V 5B loaded")

    def unload(self) -> None:
        self._model = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(
        self,
        image_bytes: bytes,
        params: dict[str, Any],
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        self._check_cancelled(cancel_event)
        if not image_bytes:
            raise RuntimeError("Wan2.2 TI2V requires an input image.")

        prompt = str(params.get("prompt") or "").strip()
        if not prompt:
            raise RuntimeError("Wan2.2 TI2V requires a non-empty prompt.")

        if self._model is None:
            _progress(progress_cb, 2, "Loading Wan2.2 TI2V")
            self.load()

        from PIL import Image
        from wan.utils.utils import save_video

        negative_prompt = str(params.get("negative_prompt") or "").strip()
        orientation = str(params.get("orientation") or "auto")
        frame_num = _as_int(params.get("duration"), 81)
        if frame_num not in {81, 121}:
            frame_num = 81 if frame_num < 101 else 121
        steps = _as_int(params.get("steps"), 20, minimum=20, maximum=50)
        guidance_scale = _as_float(params.get("guidance_scale"), 5.0, minimum=1.0, maximum=10.0)
        seed = _as_int(params.get("seed"), -1, minimum=-1, maximum=2147483647)

        _progress(progress_cb, 8, "Preparing input image")
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        target_width, target_height = _target_size_for_orientation(orientation, image.width, image.height)
        image = _cover_crop(image, target_width, target_height)

        self._check_cancelled(cancel_event)
        _progress(progress_cb, 12, "Generating video frames")
        start = time.monotonic()
        video = self._model.generate(
            prompt,
            img=image,
            size=(target_width, target_height),
            max_area=MAX_AREA_720P,
            frame_num=frame_num,
            shift=5.0,
            sample_solver="unipc",
            sampling_steps=steps,
            guide_scale=guidance_scale,
            n_prompt=negative_prompt,
            seed=seed,
            offload_model=True,
        )
        self._check_cancelled(cancel_event)

        _progress(progress_cb, 92, "Saving MP4")
        # Modly sets outputs_dir to the selected collection directory
        # (usually <workspace>/Workflows). Do not append another Workflows segment.
        output_dir = Path(self.outputs_dir) / "Wan2.2-TI2V"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        prompt_slug = _sanitize_filename(prompt)
        seed_slug = "random" if seed < 0 else str(seed)
        output_path = output_dir / f"wan22-ti2v-{stamp}-{seed_slug}-{prompt_slug}.mp4"

        save_video(
            tensor=video[None],
            save_file=str(output_path),
            fps=FPS,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        LOGGER.info("Generated %s in %.1fs", output_path, time.monotonic() - start)
        _progress(progress_cb, 100, "Done")
        return output_path
