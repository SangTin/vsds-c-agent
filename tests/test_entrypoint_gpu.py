from __future__ import annotations

from entrypoint import parse_args, resolve_gpu_layers


def test_resolve_gpu_layers_prefers_explicit_value(monkeypatch) -> None:
    monkeypatch.setenv("BANGC_N_GPU_LAYERS", "12")
    monkeypatch.setattr("entrypoint.os.path.exists", lambda path: True)
    monkeypatch.setattr("entrypoint.shutil.which", lambda command: "/usr/bin/nvidia-smi")

    assert resolve_gpu_layers(7) == 7


def test_resolve_gpu_layers_uses_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("BANGC_N_GPU_LAYERS", "12")
    monkeypatch.setattr("entrypoint.os.path.exists", lambda path: False)
    monkeypatch.setattr("entrypoint.shutil.which", lambda command: None)

    assert resolve_gpu_layers(None) == 12


def test_resolve_gpu_layers_detects_nvidia_device(monkeypatch) -> None:
    monkeypatch.delenv("BANGC_N_GPU_LAYERS", raising=False)
    monkeypatch.setattr(
        "entrypoint.os.path.exists",
        lambda path: path == "/dev/nvidia0",
    )
    monkeypatch.setattr("entrypoint.shutil.which", lambda command: None)

    assert resolve_gpu_layers(None) == 99


def test_resolve_gpu_layers_detects_nvidia_smi(monkeypatch) -> None:
    monkeypatch.delenv("BANGC_N_GPU_LAYERS", raising=False)
    monkeypatch.setattr("entrypoint.os.path.exists", lambda path: False)
    monkeypatch.setattr(
        "entrypoint.shutil.which",
        lambda command: "/usr/bin/nvidia-smi" if command == "nvidia-smi" else None,
    )

    assert resolve_gpu_layers(None) == 99


def test_resolve_gpu_layers_defaults_to_cpu(monkeypatch) -> None:
    monkeypatch.delenv("BANGC_N_GPU_LAYERS", raising=False)
    monkeypatch.setattr("entrypoint.os.path.exists", lambda path: False)
    monkeypatch.setattr("entrypoint.shutil.which", lambda command: None)

    assert resolve_gpu_layers(None) == 0


def test_n_gpu_layers_cli_uses_none_sentinel() -> None:
    assert parse_args([]).n_gpu_layers is None
    assert parse_args(["--n-gpu-layers", "4"]).n_gpu_layers == 4
