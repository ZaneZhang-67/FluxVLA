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
│   ├── config.py            # 版本化 JSON 配置及跨字段校验
│   ├── _safetensors.py      # 共享格式、SHA-256 和安全读取检查
│   ├── checkpoint.py        # 严格校验完整 state 的 key/shape/dtype
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
    ├── test_config.py
    ├── test_checkpoint.py
    └── test_fixture.py
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
    config=Path('export_config.json'),
    checkpoint=Path('model.safetensors'),
    inputs=Path('export_inputs.safetensors'),
    device='cuda:0',
)
```

```python
def prepare_tensorrt(
    *,
    recipe: str,
    config: str | Path,
    checkpoint: str | Path,
    inputs: str | Path,
    device: str,
) -> PreparedDeployment:
    ...
```

以上 `config` 参数及配置校验已实现；`pi05-v1` 是配置支持的家族协议，但尚未注册模型
recipe，因此上例还不能完成模型准备。公共入口读取版本化 JSON，recipe 收到已解析的
不可变 `DeploymentConfig`。机器人差异由配置表达，不按机器人复制模型或 recipe 实现。

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
        config: DeploymentConfig,
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
export_config.json
model.safetensors
export_inputs.safetensors
```

schema 1 显式描述模型结构、精度、推理方式、六个输入的静态契约、预处理来源，并通过
SHA-256 绑定两个 safetensors 文件。暂仅接受 FP32 master 权重、FP32/BF16 autocast
参考设置、原生完整 dense checkpoint 和普通 Euler；未合并 LoRA、RTC、外部命名格式、
FP16 执行设置及动态 fixture 在本版明确拒绝。配置解析成功不等于模型已经能够运行。

`load_config` 可独立调用，不执行 Python 配置或发现邻近文件；`load_fixture` 返回真实 CPU
输入；`load_checkpoint` 按调用方提供的完整 `expected_state` 校验 key/shape/dtype，
不修改预期 state、不改名、不隐式转换 dtype。两个读取器都验证文件格式、摘要和非有限值。
实际 PI0.5 state 的构造与完整键集合校验仍需下一阶段 recipe。

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

包骨架、公共契约、版本化部署配置、带 config 的 API 分发、真实 safetensors 读取器和
严格错误检查已实现。新增测试使用真实 JSON、safetensors 和 torch tensors；API 测试中的
临时注册类只验证分发，不代替 PI0.5 模型。当前仍没有任何内置模型 recipe。

公共 import 和配置解析不加载 torch 或训练栈。独立包测试从训练 CI 中分离，构建 wheel
后在隔离环境运行，支持 Fork 功能分支触发。下一阶段才迁移共享 PI0.5 计算代码并实现
真实参考推理；本阶段不能证明完整 checkpoint 或 ONNX/TensorRT 的数值、性能正确。
