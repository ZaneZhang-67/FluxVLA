# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import re
from pathlib import Path

from . import recipes as _recipes  # noqa: F401
from ._safetensors import validate_safetensors
from .config import load_config
from .contracts import PreparedDeployment
from .errors import ArtifactError, ContractError
from .fixture import load_fixture
from .registry import get_recipe


def prepare_tensorrt(
    *,
    recipe: str,
    config: str | Path,
    checkpoint: str | Path,
    inputs: str | Path,
    device: str,
) -> PreparedDeployment:
    """Validate deployment artifacts, then dispatch a model-family recipe.

    Preflight verifies configuration, hashes, file structure and fixture
    tensors. Only a real recipe can construct the model and validate its
    complete checkpoint state. No PI0.5 model recipe is registered yet.
    """
    if not isinstance(device, str) or not re.fullmatch(
            r'cpu|cuda(?::(?:0|[1-9][0-9]*))?', device):
        raise ContractError('device must be cpu, cuda, or cuda:<index>.')
    resolved = load_config(config, recipe=recipe)
    # Fail before reading potentially large weights when no model exists.
    implementation = get_recipe(recipe)
    checkpoint_path = validate_safetensors(
        checkpoint,
        expected_sha256=resolved.checkpoint_sha256,
        label='Checkpoint',
    )
    tensors = load_fixture(
        inputs,
        expected_sha256=resolved.inputs_sha256,
        specs=resolved.input_specs,
    )
    tokens = tensors['lang_tokens']
    if (tokens < 0).any().item() or (
            tokens >= resolved.model.vocab_size).any().item():
        raise ArtifactError(
            f'Export inputs {inputs}: lang_tokens must be in '
            f'[0, {resolved.model.vocab_size}).')
    # Recipes must revalidate hashes when loading paths. Preflight is not a
    # persistent authorization to trust files that may change afterwards.
    prepared = implementation.prepare(
        config=resolved,
        checkpoint=checkpoint_path,
        inputs=Path(inputs).expanduser().resolve(),
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
