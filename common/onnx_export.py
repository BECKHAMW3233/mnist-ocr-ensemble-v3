"""
common/onnx_export.py
======================
ONNX export — extracted verbatim from the identical export_onnx()
duplicated across every v2 training script. dynamo=False restores the
legacy TorchScript-based exporter: PyTorch 2.9+ defaults to a newer
dynamo-based exporter that requires the separate `onnxscript` package,
which this project does not otherwise need (dynamic_axes, used below, is
specifically a dynamo=False-only parameter per PyTorch's own docs).

int8 quantization note: the v2 codebase's only reference to int8
quantization was a book-chapter citation in a training script's
docstring (Chollet & Watson, "Deep Learning with Python, 3rd Ed.", Ch.
18) — no actual quantization code existed anywhere to extract. Nothing
was modularized here for it; see v3_CHANGELOG.md.
"""
from pathlib import Path

import torch
import torch.nn as nn


def export_onnx(model: nn.Module, path: str, img_size: int, validate: bool = False) -> None:
    """
    Exports a trained model to ONNX. Inference-time normalization must be
    arr = arr / 255.0 (i.e. [0,1], no mean/std shift) to match training —
    this is a contract with the callers (ocr_pipeline_mnist.py), not
    something this function enforces itself.

    validate=True additionally round-trips the exported file through
    onnx.checker.check_model() (used by the v2 SOAP scripts, which
    already depended on the `onnx` package for other reasons) — optional
    since not every training script needs the extra `onnx` package
    dependency just for this.
    """
    model.eval()
    dummy = torch.zeros(1, 1, img_size, img_size)
    torch.onnx.export(
        model, dummy, path,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    size_mb = Path(path).stat().st_size / 1024**2
    if validate:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        print(f"[ONNX] Exported and validated: {path}  ({size_mb:.1f} MB)")
    else:
        print(f"[ONNX] Exported to {path}  ({size_mb:.1f} MB)")
