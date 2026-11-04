# PI0.5 最小推理依赖与提取方案

状态：待 Review，仅源码分析，尚未提取模型或验证独立模型推理。

分析基线：`feature/fluxvla-export` 的 `3eed849`；模型源码对应上游 `cd9206a`。
本文链接固定到这两个提交，避免后续迁移导致证据行号失效。

## 结论与待确认事项

1. 使用现有通用 PyTorch PI0.5 路径，不以 ALOHA 为模型边界，也不迁移 Triton/CUDA Graph 实现。
2. 把必要计算代码迁移为 `fluxvla-export` 中的唯一实现，训练侧保留注册、构造适配和训练逻辑；不复制第二套模型。
3. 增加显式、版本化的部署配置输入。机器人差异由配置和训练环境生成的预处理 tensor 表达，不注册每个机器人的独立模型实现。
4. 首个闭环范围建议为完整原生权重、普通去噪推理；未合并 LoRA、RTC 和外部 checkpoint 格式不在首期范围，遇到时明确拒绝。

## 1. 实际推理路径

[PI05FlowMatching:63–144](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/models/vlas/pi05_flowmatching.py#L63-L144>)
继承 PI0，禁止 `state_proj`，主要重写动作与时间的 suffix embedding。
[PI0FlowMatching:828–960](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/models/vlas/pi0_flowmatching.py#L828-L960>)
提供完整预测路径：

- 视图编码：Hugging Face SigLIP → 视觉线性投影。
- prefix：视觉 tokens + 经过缩放的语言 embedding → Gemma backbone → prefix KV cache。
- denoise step：带噪动作 + timestep → 时间 MLP/AdaRMS → Gemma expert 读取 prefix cache → velocity。
- 调度：用相同 prefix cache 反复计算 velocity，执行 Euler 更新，返回归一化、padded 的动作窗口。

预测并不是调用训练时的双分支联合 attention 循环。
[forward_model:593–637](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/models/vlas/pi0_flowmatching.py#L593-L637>)
在预测时分别调用 backbone 和 expert；两路 embedding 同时存在时才走联合路径。
提取和数值对照必须覆盖真实 cached inference，不能只测试训练 forward。

## 2. 保留与排除

| 源码组件 | 处理建议 | 依据 |
| --- | --- | --- |
| PI0/PI0.5 embedding、精度转换、mask、denoise、普通 Euler 调度 | 迁移公共计算；训练侧复用 | [PI0:206–342](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/models/vlas/pi0_flowmatching.py#L206-L342>)、[预测路径](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/models/vlas/pi0_flowmatching.py#L768-L960>) |
| ConditionGemma 的 RMS/AdaRMS、MLP、RoPE、attention、decoder、model | 迁移计算类，保留 HF 模型/cache 依赖；注册和 FSDP 留在训练侧 | [ConditionGemma](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/models/backbones/llms/condition_gemma.py#L51-L1096>) |
| SigLIP 多视图 wrapper、FP32 stem 和残差行为 | 迁移；直接使用 HF SiglipVisionConfig/Model | [SigLIP:40–151](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/models/backbones/visions/siglip_vit.py#L40-L151>) |
| LinearProjector 和必要 tensor helpers | 迁移唯一实现，原导入路径转为兼容导出 | [LinearProjector:37–52](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/models/projectors/linear_projector.py#L37-L52>)、[model_utils](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/engines/utils/model_utils.py#L40-L405>) |
| BaseVLA、MMEngine builder/registry、FSDP、loss、freeze、训练采样 | 不整类迁移；推理需要的属性和构造显式化 | [BaseVLA:21–96](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/models/vlas/base_vla.py#L21-L96>)、[PI0 imports](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/models/vlas/pi0_flowmatching.py#L20-L34>) |
| PaliGemma VLM wrapper、其他模型家族、分类/生成 head | 不随 PI0.5 打包；保留训练仓库中仍有用途的代码 | PI0.5 实际由 vision、projector、两个 ConditionGemma 分别构造 |
| 自定义 atomic ops、Triton、CUDA 扩展、权重 materialization | 排除首期导出依赖 | [kernel PI0.5 imports:7–19](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/models/vlas/pi05_flowmatching_inference.py#L7-L19>) |
| tokenizer、normalizer、dataset、机器人 runner/operator | 留在训练/运行环境；导出包消费已处理的输入 | 不改变既定的 tensor fixture 边界 |

不能直接从完整 FluxVLA 导入这些类：
[顶层初始化:15–34](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/__init__.py#L15-L34>)
会导入 datasets、engines、models 等；深层路径也不会绕过父包初始化。
现有 CPU 测试通过替换 namespace 绕开这些入口，
[该辅助代码:2–10](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/test/_cpu_model_runtime.py#L2-L10>)
明确声明它不是应用 runtime，不能拿来作为独立包的实现。

### 依赖目标

拆开后，PI0.5 的直接运行依赖预计为 `torch`、`transformers==5.3.0` 和 `safetensors`。
这不是“只有三个安装包”：它们仍有传递依赖，实际锁定版本及安装闭包需在独立 wheel 测试中验证。

- 暂不重写 HF SigLIP 或 cache，以免把提取工作变成另一个模型实现。
- 不把 `mmengine`、`accelerate`、`timm`、训练数据栈和自定义 CUDA/Triton 算子列为本包直接依赖。
  SigLIP 当前的 `timm` 导入用于 FSDP wrapping，不用于视觉 forward。
- 普通公共 API 导入仍应轻量；模型依赖在实际准备 PI0.5 时加载。
- Torch 版本以现有 CPU CI 的 2.8.0 作为首个测试基线，目标机器人版本仍需独立验证；不宣称支持尚未测试的版本组合。

## 3. 一份模型代码，两侧使用

共享 core 放在独立包内，训练侧向它依赖；独立包不得反向依赖完整 `fluxvla`。
沿用包设计中的 `recipes/pi05/model/` 作为首个家族入口，共享底层组件的位置在迁移时按实际消费者拆分。

实施时需要一起满足：

- 迁移代码时移除旧计算实现，原文件只保留训练适配、注册或兼容导出。共享方法可用无参数容器的 mixin/函数复用，不维护两份 denoiser。
- 保持子模块名称与层级，例如 `llm_backbone`、`llm_expert`、`vision_backbone.vision`、`projector.projector`；不要引入 `core.*` 权重前缀。
- 不因 expert 推理没有使用 `embed_tokens` 就删掉其参数，先保持完整 checkpoint 的 key/shape 兼容。
- 训练 loss、联合 forward、FSDP policy 保留在训练侧，调用迁移后的公共数学实现。
- 迁移 SigLIP/Gemma/LinearProjector 会影响 PI0 和其他消费者，回归不能只跑 PI0.5。
- 同步训练侧安装方式和 CI：本仓库开发时安装同一 checkout 的独立包；正式发布时声明兼容版本。禁止在生产代码中补 `sys.path` 或伪造 namespace。

这会修改部分 FluxVLA 训练适配代码，目的是让训练与导出使用同一来源；不是把训练栈装进小包。

## 4. 哪些差异进入配置

| 分类 | 显式记录的内容 | 不应采用的做法 |
| --- | --- | --- |
| 模型结构 | SigLIP/Gemma/expert 的宽度、层数、head/KV-head、head_dim、vocab、激活、RoPE、norm eps、投影维度 | 从权重 shape 猜完整结构，或固定为 ALOHA 参数 |
| 推理语义 | 动作窗口、模型 padded 动作维度、去噪步数、求解方式 | 混淆 `n_action_steps` 与 `num_steps`，自动裁掉动作维度 |
| 精度策略 | 权重 dtype、autocast、FP32 flow/stem/residual、AdaRMS、protected layers、attention backend | 整个模型直接 `.half()` / `.bfloat16()` |
| tensor 边界 | batch、视图数量/顺序、图像大小、token 长度、state/noise 形状及 masks | 固定相机数量，或仅以 mask 中 True 的数量代替完整 mask |
| 外部预处理溯源 | 与 checkpoint 匹配的 prompt/tokenizer、归一化、状态/动作语义标识 | 在导出机重做或改写训练预处理 |

例如 [ALOHA 配置:139–203](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/configs/pi05/pi05_paligemma_aloha_full_finetune.py#L139-L203>)
是 50 步、原生 14 维动作、模型 padded 32 维；
[LIBERO-10:141–173](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/configs/pi05/pi05_paligemma_libero_10_full_finetune.py#L141-L173>)
配置的窗口是 10 步，模型动作同样 padded 到 32 维。
这些差异不需要第二套 PI0.5 类。

也不能仅从模型 key/shape 判断精度：
[ALOHA BF16/FP32 变体:12–28](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/configs/pi05/pi05_paligemma_aloha_bf16_expert6_fp32_full_finetune.py#L12-L28>)
保持原结构，只将前六个 expert blocks 等计算保护为 FP32；名字里的 `expert6` 不是六层 expert。

PI0.5 的数值 `states` 在 suffix 中主要提供 batch/device/dtype，并不像 PI0 一样通过 state projection 输入。
采用 state-in-prompt 的训练配置必须已把归一化的原生状态编码进 `lang_tokens`；
[PreparePromptWithState:287–328](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/transforms/prompters.py#L287-L328>)
还要求先生成 prompt，再做数值 padding。不能统一修改所有配置的 prompt，或以替换 `states` tensor 代替更新 prompt。

### 建议的输入边界

首个 reference adapter 保持现有标准模型的输入，不擅自删除 `states`：

```text
images       [B, 3*V, H, W]  浮点，已按训练配置处理
img_masks    [B, V]          bool，True 表示有效视图
lang_tokens  [B, L]          int64，已完成 prompt/tokenizer
lang_masks   [B, L]          bool，True 表示有效 token
states       [B, S]          浮点，dtype 与配置一致
noise        [B, T, A]       float32，显式固定且与窗口/模型动作维度匹配
```

按配置验证同一 batch、view/token 数量和取值范围；`action_masks` 是训练监督输入，不加入普通预测契约。
未来导出 wrapper 若移除未参与计算的输入，必须单独验证与完整 reference 等价。

### 公共 API 的必要补充（提案，未实现）

[当前 API:29–35](<https://github.com/ZaneZhang-67/FluxVLA/blob/3eed849/packages/fluxvla-export/src/fluxvla_export/api.py#L29-L35>)
没有配置输入。建议给公共入口和 `ExportRecipe.prepare` 增加必填 `config` 参数：

```python
prepared = prepare_tensorrt(
    recipe='pi05-v1',
    config='export_config.json',
    checkpoint='model.safetensors',
    inputs='export_inputs.safetensors',
    device='cuda:0',
)
```

这里的 recipe 表示模型家族与协议版本，不表示机器人平台。示例 recipe 和配置文件均尚未实现。
配置由训练环境导出为解析后的、版本化 JSON；构建机不执行 MMEngine Python 配置，不自动找相邻文件。
配置须标明原生 checkpoint 格式、架构、精度和输入契约，并用 checkpoint/fixture 摘要绑定三者。
缺失、未知字段或不支持的组合直接报错。仅校验 shape 无法发现相机顺序或归一化错误。

输出先保留模型原有的完整 padded 动作窗口；原生动作裁剪、反归一化和机器人动作转换仍由外部运行系统处理。
这点与现有预测的 `return x_t` 一致，与训练 loss 使用的动作裁剪不是同一边界。

## 5. 提取时必须守住的数值行为

- 显式固定 noise，reference 每次克隆它。现有 Euler 是 in-place 更新，直接复用会让第二次比较从错误状态开始。
- backbone 更新 prefix cache；expert 去噪时只拼接该 cache，不把 suffix 写回它。
  依据：[GemmaAttention:558–574](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/fluxvla/models/backbones/llms/condition_gemma.py#L558-L574>)。
- 验证 prefix/expert 的层数、KV-head 和 head_dim 兼容；保留 masks、位置编码及有限 BF16-min mask sentinel，不能随意替换为 `-inf`。
- 不用训练预加载器默认的非 strict 行为。完整原生权重必须严格加载全部 key/shape；外部 OpenPI 命名映射与 LoRA 合并应另设明确转换步骤。
- 保留 protected FP32 运算；FP16/BF16 加速的数值接受标准在目标机验证，不能把“开启半精度”当作全部权重和计算都为半精度。
- RTC prefix/guidance 是推理能力，不是纯训练依赖。首期暂不支持时必须检查配置并拒绝，不能静默退化为普通采样。

以后分图可沿 prefix prefill 和单个 denoise step 的自然边界进行，cache 显式转成 tensor I/O，Euler 循环由 executor 执行。
这只是后续候选分图；HF cache 对象、attention backend、导出支持和 TensorRT 性能仍须实测，不在本块承诺可导出。

## 6. 分块实施与验收

| 块 | 内容 | 验收 |
| --- | --- | --- |
| 下一块 | 配置 schema/API 分发、checkpoint/fixture 的真实 safetensors 读取与严格错误检查 | 能解析有效输入；缺失/错配/不支持配置可定位地失败；不生成假模型结果 |
| 共享 core 迁移 | 先留存旧实现的数值基线，再迁移公共计算并改训练适配 | 真实小模型参数 key/shape、prefix、单步 velocity、最终 action 与迁移前对照；训练 forward/backward 回归 |
| 独立包闭环 | 新环境仅安装 wheel 和推理依赖，用真实 nn.Module/Tensor、实际 safetensors 做参考推理 | 不借助 FluxVLA namespace 或完整仓库；支持不同视图数/窗口/维度配置；重复运行不改 fixture/RNG |
| 真实权重及加速 | 使用与实际 checkpoint 配套的配置和输入，之后接 ONNX/TensorRT | 真实目标设备上的输出误差、性能、日志及构建产物 |

可复用的测试基础已经存在：

- [tiny PI0/PI0.5 测试:243–348](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/test/test_models/test_vla.py#L243-L348>)：真实缩小版 SigLIP、Gemma、expert，B=2、不完整 masks、固定 noise、训练梯度和 cached inference。
- [cache/joint 对照:106–150](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/test/test_models/test_pi05_attention_precision.py#L106-L150>)：真实 attention 在 FP32/BF16 下的缓存路径对照。
- [精度保护测试:200–292](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/test/test_models/test_pi05_attention_precision.py#L200-L292>)：缩小后的真实配置保留六个 protected blocks 和一个 BF16 block，验证梯度、普通和 RTC 推理。

这些是现有测试源码证据，本次没有执行它们。当前包的 19 项骨架测试也不能替代真实 tensor、模型加载或导出验证。
另有[精度测试:174–197](<https://github.com/FluxVLA/FluxVLA/blob/cd9206a/test/test_models/test_pi05_attention_precision.py#L174-L197>)
引用了当前配置目录不存在的两个 basket 配置；运行基线时需单独记录此已有问题，不得把它当作本次提取回归或直接宣称整套测试通过。

每块完成后先 Review，再 commit/push 到 Fork 功能分支。真实训练 checkpoint 及配套配置在真实权重验证块准备，现在不需要选择具体机器人。
