"""Smoke tests for configuration and package imports (no model download)."""

from __future__ import annotations

import importlib

import pytest


def test_settings_paths_are_absolute():
    from pitgpt.config import settings

    assert settings.paths.root.is_absolute()
    assert settings.model.base_model_id  # non-empty
    assert settings.lora.r > 0
    assert settings.training.gradient_accumulation_steps >= 1


def test_effective_batch_size_is_sixteen():
    from pitgpt.config import settings

    eff = (
        settings.training.per_device_train_batch_size
        * settings.training.gradient_accumulation_steps
    )
    assert eff == 16, f"effective batch changed to {eff}"


def test_lora_targets_are_llama_style():
    from pitgpt.config import settings

    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert proj in settings.lora.target_modules


@pytest.mark.parametrize(
    "module",
    [
        "pitgpt.config.settings",
        "pitgpt.data.build_dataset",
        "pitgpt.data.validate_dataset",
        "pitgpt.evaluation.prompts",
        "pitgpt.inference.engine",
        "pitgpt.inference.model_loader",
        "pitgpt.training.data_module",
        "pitgpt.training.train",
        "pitgpt.app.model_service",
        "pitgpt.app.theme",
    ],
)
def test_modules_import(module):
    """Every module imports without side effects (no model load at import time)."""
    importlib.import_module(module)


def test_eval_prompt_set_is_twenty():
    from pitgpt.evaluation.prompts import EVAL_PROMPTS

    assert len(EVAL_PROMPTS) == 20
    assert all(isinstance(p, str) and p.strip() for p in EVAL_PROMPTS)
