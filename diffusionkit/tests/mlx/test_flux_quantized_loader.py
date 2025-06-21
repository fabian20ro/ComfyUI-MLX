import types
import pytest

mx = pytest.importorskip("mlx.core")
import diffusionkit.mlx.model_io as model_io

class DummyModel:
    def update(self, *args, **kwargs):
        pass

def dummy_quantize(model):
    return model

def dummy_hf_download(*args, **kwargs):
    return "/tmp/dummy"

def dummy_mx_load(path):
    return {"adaLN.weight": mx.array([1, 2, 3, 4], dtype=mx.float32)}


def test_flux_quantized_weights_are_uint32(monkeypatch):
    monkeypatch.setattr(model_io, "MMDiT", lambda cfg: DummyModel())
    monkeypatch.setattr(model_io.nn, "quantize", dummy_quantize)
    monkeypatch.setattr(model_io, "hf_hub_download", dummy_hf_download)
    monkeypatch.setattr(model_io.mx, "load", dummy_mx_load)

    model_key = "argmaxinc/mlx-FLUX.1-schnell-4bit-quantized"
    flat, _ = model_io.load_flux(
        key=model_key,
        model_key=model_key,
        only_modulation_dict=True,
    )
    assert all(w.dtype == mx.uint32 for w in flat)

