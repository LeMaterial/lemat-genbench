"""Tests that MACE-MP loads the checkpoints its references were built with.

The ``mace_mp`` hull (LeMaterial/LeMat-Bulk-MLIP-Hull) and the Fréchet distance
reference (LeMaterial/LeMat-GenBench-embeddings) come from two different
MACE-MP checkpoints, so energies and embeddings must each use their own.
"""

import pytest

pytest.importorskip("mace.calculators")

from lemat_genbench.models.mace import calculator as mace_calculator  # noqa: E402


class _FakeASECalculator:
    def __init__(self, model):
        self.model = model


@pytest.fixture
def fake_mace_mp(monkeypatch):
    calls = []

    def fake(model=None, device="", **kwargs):
        calls.append({"model": model, "device": device, **kwargs})
        return _FakeASECalculator(model)

    monkeypatch.setattr(mace_calculator, "mace_mp", fake)
    return calls


def test_energy_and_embedding_checkpoints_are_pinned():
    assert mace_calculator.MACE_MP_ENERGY_MODEL == "medium-0b3"
    assert mace_calculator.MACE_MP_EMBEDDING_MODEL == "medium-mpa-0"


def test_mp_calculator_uses_hull_checkpoint_for_energies(fake_mace_mp):
    calc = mace_calculator.MACECalculator(model_type="mp", hull_type="mace_mp")

    assert calc._get_ase_calculator().model == "medium-0b3"
    assert calc.embedding_extractor.calculator.model == "medium-mpa-0"
    assert [call["model"] for call in fake_mace_mp] == ["medium-0b3", "medium-mpa-0"]


def test_shared_checkpoint_is_loaded_once(fake_mace_mp):
    calc = mace_calculator.MACECalculator(
        model_type="mp", model="medium-mpa-0", embedding_model="medium-mpa-0"
    )

    assert len(fake_mace_mp) == 1
    assert calc.embedding_extractor.calculator is calc._get_ase_calculator()


def test_fallback_loading_keeps_the_checkpoints(monkeypatch):
    calls = []

    def flaky(model=None, device="", **kwargs):
        calls.append(model)
        if len(calls) == 1:
            raise RuntimeError("first load fails")
        return _FakeASECalculator(model)

    monkeypatch.setattr(mace_calculator, "mace_mp", flaky)
    calc = mace_calculator.MACECalculator(model_type="mp")

    assert calc._get_ase_calculator().model == "medium-0b3"
    assert calc.embedding_extractor.calculator.model == "medium-mpa-0"
    assert "small" not in calls


def test_checkpoint_aliases_exist_in_installed_mace():
    """Both aliases must resolve in mace-torch rather than fall back to its default."""
    import inspect

    from mace.calculators import foundations_models

    source = inspect.getsource(foundations_models.download_mace_mp_checkpoint)
    registry = getattr(foundations_models, "mace_mp_urls", None)
    for alias in ("medium-0b3", "medium-mpa-0"):
        assert (registry is not None and alias in registry) or f'"{alias}"' in source
