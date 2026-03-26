from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent.parent


def _resolve_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


SAM3_REPO = Path(os.environ.get("NAV_SAM3_REPO", str(ROOT_DIR / "sam3_main")))
_sam3_ckpt_candidates = []
if "NAV_SAM3_CKPT" in os.environ:
    _sam3_ckpt_candidates.append(Path(os.environ["NAV_SAM3_CKPT"]))
_sam3_ckpt_candidates.extend(
    [
        ROOT_DIR / "sam3.pt",
        SAM3_REPO / "sam3.pt",
    ]
)
SAM3_CKPT = _resolve_existing_path(_sam3_ckpt_candidates)


def _encode_error(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _normalize_box_xyxy(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) * 0.5) / max(1, width)
    cy = ((y1 + y2) * 0.5) / max(1, height)
    bw = max(0.0, x2 - x1) / max(1, width)
    bh = max(0.0, y2 - y1) / max(1, height)
    return [cx, cy, bw, bh]


def main() -> None:
    if not SAM3_REPO.exists():
        sys.stdout.write(json.dumps(_encode_error(f"sam3_repo_missing: {SAM3_REPO}")) + "\n")
        sys.stdout.flush()
        return
    if SAM3_CKPT is None:
        sys.stdout.write(
            json.dumps(
                _encode_error(
                    "sam3_ckpt_missing: expected NAV_SAM3_CKPT or one of "
                    f"{ROOT_DIR / 'sam3.pt'} / {SAM3_REPO / 'sam3.pt'}"
                )
            )
            + "\n"
        )
        sys.stdout.flush()
        return
    try:
        sys.path.insert(0, str(SAM3_REPO))
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as exc:
        sys.stdout.write(json.dumps(_encode_error(f"sam3_import_failed: {exc}")) + "\n")
        sys.stdout.flush()
        return

    requested_device = os.environ.get("NAV_SAM3_DEVICE", "").strip()
    device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        sys.stdout.write(
            json.dumps(
                _encode_error(
                    "sam3_cuda_unavailable: torch.cuda.is_available() is False; "
                    "set NAV_SAM3_PYTHON to a GPU-enabled environment or fix GPU/driver visibility"
                )
            )
            + "\n"
        )
        sys.stdout.flush()
        return
    if device.startswith("cuda") and not torch.cuda.is_available():
        sys.stdout.write(
            json.dumps(
                _encode_error(
                    "sam3_cuda_unavailable: requested CUDA but torch.cuda.is_available() is False"
                )
            )
            + "\n"
        )
        sys.stdout.flush()
        return
    try:
        model = build_sam3_image_model(
            checkpoint_path=str(SAM3_CKPT),
            load_from_HF=False,
            device=device,
            eval_mode=True,
        )
        processor = Sam3Processor(model, device=device, confidence_threshold=0.35)
    except Exception as exc:
        sys.stdout.write(json.dumps(_encode_error(f"sam3_init_failed: {exc}")) + "\n")
        sys.stdout.flush()
        return

    sys.stdout.write(
        json.dumps(
            {
                "ok": True,
                "ready": True,
                "device": device,
                "checkpoint_path": str(SAM3_CKPT),
                "repo_path": str(SAM3_REPO),
            }
        )
        + "\n"
    )
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            raw = base64.b64decode(request["image_base64_png"])
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            state = processor.set_image(image)
            text_prompt = request.get("text_prompt")
            bbox = request.get("bbox")
            if text_prompt:
                state = processor.set_text_prompt(prompt=text_prompt, state=state)
            if bbox is not None:
                state = processor.add_geometric_prompt(
                    box=_normalize_box_xyxy(bbox, image.width, image.height),
                    label=True,
                    state=state,
                )

            scores = state.get("scores")
            boxes = state.get("boxes")
            masks = state.get("masks")
            if scores is None or boxes is None or masks is None or len(scores) == 0:
                response = {"ok": True, "detections": []}
            else:
                best_idx = int(torch.argmax(scores).item())
                best_score = float(scores[best_idx].item())
                best_box = [float(v) for v in boxes[best_idx].detach().cpu().tolist()]
                best_mask = masks[best_idx].detach().cpu().numpy().astype(np.uint8)
                response = {
                    "ok": True,
                    "detections": [
                        {
                            "score": best_score,
                            "bbox_xyxy": best_box,
                            "area_pixels": int(best_mask.sum()),
                            "center_px": [
                                int(round((best_box[0] + best_box[2]) * 0.5)),
                                int(round((best_box[1] + best_box[3]) * 0.5)),
                            ],
                        }
                    ],
                }
        except Exception as exc:
            response = _encode_error(f"sam3_inference_failed: {exc}")
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
