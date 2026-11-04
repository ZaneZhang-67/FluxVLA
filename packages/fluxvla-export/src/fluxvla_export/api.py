# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from pathlib import Path

from . import recipes as _recipes  # noqa: F401
from .contracts import PreparedDeployment
from .errors import ArtifactError, ContractError
from .registry import get_recipe


def _safetensors_file(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ArtifactError(f'{label} safetensors file not found: {path}')
    if path.suffix != '.safetensors':
        raise ArtifactError(
            f'{label} must use the .safetensors format: {path}')
    return path


def prepare_tensorrt(
    *,
    recipe: str,
    checkpoint: str | Path,
    inputs: str | Path,
    device: str,
) -> PreparedDeployment:
    """Prepare one FluxVLA checkpoint for the FluxHermes build pipeline."""
    if (
        not isinstance(device, str)
        or not device
        or device.strip() != device
    ):
        raise ContractError(
            'device must be a non-empty string without surrounding spaces.')
    checkpoint_path = _safetensors_file(checkpoint, label='Checkpoint')
    input_path = _safetensors_file(inputs, label='Export inputs')
    implementation = get_recipe(recipe)
    prepared = implementation.prepare(
        checkpoint=checkpoint_path,
        inputs=input_path,
        device=device,
    )
    if not isinstance(prepared, PreparedDeployment):
        raise ContractError(
            f'Recipe {recipe!r} must return PreparedDeployment, got '
            f'{type(prepared).__name__}.')
    if prepared.recipe != recipe:
        raise ContractError(
            f'Recipe {recipe!r} returned deployment for '
            f'{prepared.recipe!r}.')
    return prepared
