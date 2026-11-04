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
│   ├── components/          # 已迁移的共享底层计算；训练也使用这一份
│   │   ├── __init__.py      # 不触发模型依赖加载
│   │   ├── condition_gemma.py
│   │   ├── siglip.py
│   │   ├── projector.py
│   │   └── tensor_utils.py
│   └── recipes/
│       └── pi05/            # 后续实现，当前尚未注册
│           ├── recipe.py    # PI0.5 构造入口
│           ├── model/       # PI0.5 policy 计算；复用 components
│           ├── graphs.py    # 子图拆分和 export wrapper
│           ├── reference.py # 确定性 PyTorch reference
│           └── executor.py  # 多引擎执行顺序
└── tests/
    ├── test_api.py
    ├── test_contracts.py
    ├── test_registry.py
    ├── test_config.py
    ├── test_checkpoint.py
    ├── test_fixture.py
    ├── test_shared_gemma.py
    ├── test_shared_vision_projector.py
    └── test_shared_tensor_utils.py
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

公共 import、components 命名空间和配置解析不加载 torch 或训练栈。独立包测试从训练
CI 中分离，构建 wheel 后在隔离环境运行，支持 Fork 功能分支触发。

本块已迁移 ConditionGemma、SigLIP 多视图与 FP32 stem、LinearProjector 和十个纯 tensor
helper。训练侧从独立包继承或直接兼容导出，没有保留第二份计算；registry/FSDP、视觉 ID
及 pretrained 解析、语言生成/分类 head 留在训练侧。原来的参数层级保持不变，未引入
`core.*` 前缀。旧属性和未使用 head 中与此次迁移无关的既有问题未顺带修改。

独立包模型依赖为 Transformers 5.3.0；Torch 2.6 与 2.8 均作为 CPU 组件测试基线。
下界保留 2.6 是为了兼容原训练安装脚本，而不是让安装共享包隐式升级 CUDA Torch。
训练安装/更新脚本和手工说明会先安装同一 checkout 的独立包，再安装根包；根包明确
声明对 `fluxvla-export==0.1.0` 的依赖，不通过修改生产 `sys.path` 寻找源码。

PI0/PI0.5 的 embedding、denoising、Euler 与训练 loss 暂仍在训练仓库。下一块再迁移
policy 层公共计算，并接独立参考推理；本块没有实现 PI0.5 recipe 或 ONNX/TensorRT。

## 底层迁移的数值验证

原始源码固定在提交 `0439d4a2e61abc2ecbb03c38b08e27d59f27c369` 的独立 worktree。
测试探针先从该原始版本生成真实 safetensors 参数、输入、结果和梯度，再在同一 Python /
Torch / Transformers 环境下，与迁移版本进行独立进程比较；没有在代码里编造 golden
outputs 或保留可供生产调用的 legacy 模型副本。

- 原版 24 个 case 全部运行成功；迁移后 training/direct-shared 合计 40 个比较全部通过。
- `rtol=0`、`atol=0`，所有已比较 tensor 的最大绝对差均为 0。
- 包括 PI0、PI0.5 的 FP32/BF16 与普通/保护精度组合：完整初始化参数及 key/shape/dtype、
  训练 loss/velocity/梯度、prefix embedding/hidden/cache、单步 velocity、最终 action。
- 使用 B=2、不完整 masks 和显式 noise；每次 Euler 都克隆 noise，并检查 fixture/RNG 不变。
- 原 CPU 回归清单 191 项通过；新增 43 项复用检查后，234 项在三个 seed/order 下全部通过。
  这仍是仓库明确划定的 CPU 单元测试，不是完整 GPU/FSDP 集成测试。
- wheel 从 sdist 构建，在两个全新环境分别使用 Torch 2.6.0 / 2.8.0，每个环境 133 项
  测试通过、零跳过；未安装 FluxVLA、MMEngine、timm、accelerate、torchvision 或 Triton。
  已核对安装包的全部 Python 源码与当前实现一致，公共 import 仍不加载模型依赖。

重现原版与迁移版对照（从当前仓库运行，输出目录必须尚不存在）：

```bash
git worktree add --detach /tmp/fluxvla-before-components 0439d4a
python test/export_components_probe.py capture \
  --repo /tmp/fluxvla-before-components --expected-commit 0439d4a \
  --output /tmp/component-baseline
python test/export_components_probe.py compare \
  --repo . --baseline /tmp/component-baseline \
  --output /tmp/component-comparison
```

探针使用既有 CPU 测试 bootstrap 导入真实训练模块；独立 wheel 的组件测试不使用它。
在本机 managed Python 3.12 的 venv 下，MMEngine 0.10.7 会把 stdlib 的 `os` / `copy`
误判成外部依赖，导致原版也出现 9 个配置测试错误。完整 CPU 清单验证改用同一个真实
base interpreter、`-X frozen_modules=off`，并通过测试进程的 `PYTHONPATH` 引入隔离
环境的依赖；检测真实 stdlib 来源通过后，原版和迁移版均通过。没有 patch MMEngine、
mock 配置或跳过失败测试。Ubuntu/Python 3.10 CI 保留原有 stdlib 检测门槛。

上述数值结果只覆盖当前 CPU 小模型。真实训练 checkpoint、目标 GPU 精度、FSDP、
ONNX/TensorRT 支持和性能仍需后续独立验证。
