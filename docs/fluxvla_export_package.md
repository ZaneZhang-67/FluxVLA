# `fluxvla-export` 最小包设计

## 目标

`fluxvla-export` 是 FluxVLA 面向 FluxHermes 的构建前端：把受支持的
FluxVLA checkpoint 准备成可导出的 PyTorch 子图和确定性参考推理。

职责边界：

```text
fluxvla-export                         FluxHermes
模型构造与 safetensors 加载            ONNX 导出
模型家族分图与输入校验          ───>   TensorRT 构建
PyTorch reference 与引擎编排           parity、benchmark、bundle
```

## 约束

- 独立发布 wheel，分发名为 `fluxvla-export`，导入名为 `fluxvla_export`。
- 首期只支持 PI0.5；通过后再增加 GR00T-N1.5/N1.7。
- 不依赖完整 `fluxvla` 包，避免安装训练、数据集、仿真和机器人依赖。
- 不依赖 FluxHermes、ONNX 或 TensorRT；依赖方向保持为
  `FluxHermes -> fluxvla-export`。
- 只接受 safetensors，不支持 pickle checkpoint。
- 输入是已经预处理的 tensor fixture；包内不包含数据集、tokenizer 和归一化流程。
- 缺少 recipe、输入、权重或版本不匹配时直接报错，不做自动猜测或降级。

## 目录

```text
packages/fluxvla-export/
├── LICENSE
├── pyproject.toml
├── README.md
├── src/fluxvla_export/
│   ├── __init__.py          # 仅导出稳定公共 API
│   ├── py.typed             # PEP 561 类型标记
│   ├── api.py               # prepare_tensorrt
│   ├── contracts.py         # PreparedDeployment/PreparedGraph/TensorSpec
│   ├── checkpoint.py        # 安全加载 safetensors、严格校验 key/shape
│   ├── fixture.py           # 加载预处理输入并校验契约
│   ├── errors.py            # 可定位的公开异常
│   ├── registry.py          # 包内静态 recipe 注册，不做第三方插件发现
│   └── recipes/
│       └── pi05/            # 第二阶段实现
│           ├── recipe.py    # PI0.5 构造入口
│           ├── model/       # PI0.5 推理所需的最小模型代码
│           ├── graphs.py    # 子图拆分和 export wrapper
│           ├── reference.py # 确定性 PyTorch reference
│           └── executor.py  # 多引擎执行顺序
└── tests/
    ├── test_api.py
    ├── test_contracts.py
    ├── test_registry.py
    ├── test_checkpoint.py     # PI0.5 实现阶段增加
    └── test_fixture.py        # PI0.5 实现阶段增加
```

不得通过复制整个 `fluxvla/` 或依赖根包来快速完成该目录；PI0.5 实现阶段应按真实
import 链提取最小推理依赖。公共计算迁移为独立包中的唯一实现，原训练代码通过适配复用，
不能在两个目录分别维护 PI0.5 模型。

## 唯一公共入口

```python
from pathlib import Path

from fluxvla_export import prepare_tensorrt

prepared = prepare_tensorrt(
    recipe='pi05-v1',
    checkpoint=Path('model.safetensors'),
    inputs=Path('export_inputs.safetensors'),
    device='cuda:0',
)
```

```python
def prepare_tensorrt(
    *,
    recipe: str,
    checkpoint: str | Path,
    inputs: str | Path,
    device: str,
) -> PreparedDeployment:
    ...
```

以上签名是当前骨架接口，`pi05-v1` 仅为未注册的家族 recipe 示例，尚不能运行。
PI0.5 依赖分析后建议增加必填 `config` 参数，显式接收与 checkpoint 对应的版本化部署配置；
该变更待 Review，尚未实现。机器人差异由配置表达，不按机器人复制模型或 recipe 实现。

所有参数显式传入。checkpoint 与 recipe 的自动发现由上层 artifact manifest 负责，
不放在本包首期范围内。

## 返回契约

```python
@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: tuple[int, ...]
    min_shape: tuple[int, ...] | None = None
    opt_shape: tuple[int, ...] | None = None
    max_shape: tuple[int, ...] | None = None


@dataclass
class PreparedGraph:
    name: str
    module: torch.nn.Module
    inputs: Mapping[str, torch.Tensor]
    input_specs: tuple[TensorSpec, ...]
    output_specs: tuple[TensorSpec, ...]
    export_options: Mapping[str, object]
    build_options: Mapping[str, object]


class InferenceEngine(Protocol):
    def infer(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        synchronize: bool = False,
    ) -> Mapping[str, torch.Tensor]: ...


class DeploymentExecutor(Protocol):
    def infer(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        synchronize: bool = False,
    ) -> Mapping[str, torch.Tensor]: ...


@dataclass
class PreparedDeployment:
    recipe: str
    graphs: Mapping[str, PreparedGraph]
    policy_inputs: Mapping[str, torch.Tensor]
    policy_input_specs: tuple[TensorSpec, ...]
    output_specs: tuple[TensorSpec, ...]
    reference_runner: Callable[[], Mapping[str, torch.Tensor]]
    executor_factory: Callable[
        [Mapping[str, InferenceEngine]], DeploymentExecutor]
    metadata: Mapping[str, object]

    def run_reference(self) -> Mapping[str, torch.Tensor]: ...

    def create_executor(
        self,
        engines: Mapping[str, InferenceEngine],
    ) -> DeploymentExecutor: ...


class ExportRecipe(Protocol):
    name: str

    def prepare(
        self,
        *,
        checkpoint: Path,
        inputs: Path,
        device: str,
    ) -> PreparedDeployment: ...
```

FluxHermes 只依赖以上契约，不导入 PI0.5 内部模块。构建期 executor 留在本包，
因为子图顺序、cache 传递和 denoise 循环属于模型语义。bundle 运行期是否继续依赖
本包，在 PI0.5 闭环后单独决定，本阶段不设计通用执行 DSL。

## 输入 artifact

首期由训练环境生成，目标构建机只消费：

```text
model.safetensors
export_inputs.safetensors
```

`export_inputs.safetensors` 保存已经完成 tokenizer、图像处理、状态归一化和 padding
的 tensor，并显式包含用于 parity 的固定 noise。它不包含原始数据或 pickle。

recipe 负责严格验证：

- 输入名称必须完全匹配；
- dtype 和 shape 必须匹配；
- checkpoint key 和 shape 必须严格匹配；
- 不接受未知输入或缺失输入。

## 不包含

- 训练 runner、optimizer、scheduler、DDP/FSDP；
- dataset、collator、tokenizer、图像和状态预处理；
- ROS、机器人 operator、仿真和评测；
- ONNX、TensorRT builder/runtime、bundle；
- INT8 校准；
- 自动模型识别、entry point 插件系统或通用执行 DSL。

## 当前实现状态

包骨架、公共契约、静态 recipe registry、API 分发和基础测试已经实现；当前没有
注册任何内置 recipe。下一阶段才提取 PI0.5 最小推理依赖并实现首个 recipe。
