from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from fluxvla_export import (ArtifactError, ContractError, PreparedDeployment,
                            PreparedGraph, TensorSpec, prepare_tensorrt)
from fluxvla_export.registry import register_recipe


class FakeTensor:

    def __init__(self, shape, dtype='float32'):
        self.shape = shape
        self.dtype = dtype

    def detach(self):
        return self


class FakeModule:

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def forward(self, *inputs):
        return inputs[0]

    def eval(self):
        return self

    def state_dict(self):
        return {}


class DummyExecutor:

    def infer(self, inputs, *, synchronize=False):
        return {'actions': inputs['images']}


class ApiTest(unittest.TestCase):

    @staticmethod
    def deployment(recipe_name):
        image_spec = TensorSpec('images', 'float32', (1, 3, 224, 224))
        graph = PreparedGraph(
            name='model',
            module=FakeModule(),
            inputs={'images': FakeTensor(image_spec.shape)},
            input_specs=(image_spec, ),
            output_specs=(TensorSpec(
                'actions', 'float32', (1, 10, 7)), ),
        )
        return PreparedDeployment(
            recipe=recipe_name,
            graphs={'model': graph},
            policy_inputs={'images': FakeTensor(image_spec.shape)},
            policy_input_specs=(image_spec, ),
            output_specs=(TensorSpec(
                'actions', 'float32', (1, 10, 7)), ),
            reference_runner=lambda: {
                'actions': FakeTensor((1, 10, 7)),
            },
            executor_factory=lambda engines: DummyExecutor(),
        )

    def test_dispatches_explicit_recipe(self):
        recipe_name = f'test-{uuid.uuid4().hex}'
        prepared = self.deployment(recipe_name)

        class Recipe:
            name = recipe_name

            def prepare(self, *, checkpoint, inputs, device):
                self.arguments = (checkpoint, inputs, device)
                return prepared

        register_recipe(recipe_name)(Recipe)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'model.safetensors'
            inputs = root / 'inputs.safetensors'
            checkpoint.write_bytes(b'checkpoint')
            inputs.write_bytes(b'inputs')
            result = prepare_tensorrt(
                recipe=recipe_name,
                checkpoint=checkpoint,
                inputs=inputs,
                device='cuda:0',
            )
        self.assertIs(result, prepared)

    def test_rejects_non_safetensors_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'model.pt'
            inputs = root / 'inputs.safetensors'
            checkpoint.write_bytes(b'checkpoint')
            inputs.write_bytes(b'inputs')
            with self.assertRaisesRegex(ArtifactError, '.safetensors format'):
                prepare_tensorrt(
                    recipe='unused',
                    checkpoint=checkpoint,
                    inputs=inputs,
                    device='cuda:0',
                )

    def test_rejects_missing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'inputs.safetensors'
            inputs.write_bytes(b'inputs')
            with self.assertRaisesRegex(ArtifactError, 'not found'):
                prepare_tensorrt(
                    recipe='unused',
                    checkpoint=root / 'missing.safetensors',
                    inputs=inputs,
                    device='cuda:0',
                )

    def test_rejects_invalid_device(self):
        for device in ('', ' cuda:0', 'cuda:0 ', 0):
            with self.subTest(device=device):
                with self.assertRaisesRegex(ContractError, 'device'):
                    prepare_tensorrt(
                        recipe='unused',
                        checkpoint='unused.safetensors',
                        inputs='unused.safetensors',
                        device=device,
                    )


if __name__ == '__main__':
    unittest.main()
