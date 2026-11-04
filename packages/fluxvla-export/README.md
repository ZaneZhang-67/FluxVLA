# fluxvla-export

Minimal FluxVLA build frontend consumed by FluxHermes.

The package owns model construction, safetensors loading, export graph
partitioning, deterministic PyTorch reference inference, and model-specific
engine orchestration. FluxHermes owns ONNX export, TensorRT building,
validation, benchmarking, and packaging.

No model recipe is shipped in the initial package skeleton. Calling
`prepare_tensorrt` with an unregistered recipe raises `RecipeNotFoundError`.
PI0.5 will be the first built-in recipe after the public contracts are
reviewed.
