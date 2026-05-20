# DeepGEMM

DeepGEMM for PPU 是基于 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 适配 PPU 硬件特性的版本，支持 DeepGEMM 在 PPU 上的计算执行。


## 快速开始

### 环境要求

- ZW 610 / 610E / 810 / 810E / M530 / M890 / 890L / 890P
- Python 3.8 或更高版本
- PPU SDK 12.3 或更高版本
- PyTorch 2.1 或更高版本
- CUTLASS3 for PPU: v3.6.0_release

### 开发流程

```bash
# 必须递归克隆子模块
git clone --recursive git@gitlab.alibaba-inc.com:ppu_open_source/DeepGemm.git

# 为第三方头文件目录（CUTLASS 和 CuTe）创建符号链接
python setup.py develop

# 测试 JIT 编译
python tests/test_jit.py

# 测试所有 GEMM 实现（普通、连续分组、掩码分组）
python tests/test_core.py
```

### 安装方式

```bash
python setup.py install
```

安装完成后，在您的 Python 项目中导入 `deep_gemm` 即可使用！

## 接口说明

#### 注意事项

本库仅包含 GEMM 内核。它要求左侧矩阵（LHS）的缩放因子满足 AIU 对齐并进行转置，且仅支持 NT 格式（左侧非转置，右侧转置）。如需执行转置或其他 FP8 类型转换操作，请自行实现或将这些操作融合到前置内核中。虽然本库提供了一些简单的 PyTorch 工具函数，但这些函数可能性能较低；我们的主要优化目标始终是 GEMM 内核本身。

#### 数据精度

| 算法 | INT8 | FP4 | FP8 | BF16 |
| :--- | :---: | :---: | :---: | :---: |
| **Non-grouped** | ✅ | ✅ | ✅ | ✅ |
| **Contiguous** | ✅ | ✅ | ✅ | ✅ |
| **No-pad** | ✅ | ✅ | ✅ | ✅ |
| **Masked** | ✅ | ✅ | ✅ | ✅ |

#### 常规稠密 GEMM（非 Grouped）

执行基本的非 Group FP8 GEMM，请调用 `deep_gemm.gemm_fp8_fp8_bf16_nt` 函数。详细信息请参考该函数的文档。

#### Grouped GEMM（contiguous layout）

与 CUTLASS 中的传统 Grouped GEMM 不同，DeepGEMM 仅对 M 轴进行分组，而 N 和 K 维度必须保持固定。这种设计特别适用于 MoE 模型中各专家具有相同形状的场景。

在训练前向传播或推理 prefill 阶段，当每个专家处理的 token 数量不同时，我们将这些 token 拼接成一个单一张量，称为“contiguous layout”。请注意，每个专家的数据段必须对齐到 GEMM 的 M 块大小（可通过 `get_m_alignment_for_contiguous_layout()` 获取）。

更多信息请参阅 `m_grouped_gemm_fp8_fp8_bf16_nt_contiguous` 函数的文档。

#### Grouped GEMM（nopad layout）
我们提供了一种新的 Grouped GEMM 模式，即nopad layout。当每个专家处理的 token 数量不同时，通过提供索引张量存储不同专家处理的token数量，无需对齐到 GEMM 的 M 块大小，减少无效的数据搬运.

更多信息请参阅 `m_grouped_gemm_fp8_fp8_bf16_nt_nopad` 函数的文档。

#### Grouped GEMM（masked layout）

在推理解码阶段，若启用了 HGGC Graph 且 CPU 无法预知每个专家接收到的 token 数量，我们支持带掩码的 Grouped GEMM。通过提供一个掩码张量，内核将仅计算有效部分。

请使用 `m_grouped_gemm_fp8_fp8_bf16_nt_masked` 函数，并查阅相关文档。一个典型用例是将其输入设为来自 [DeepEP](https://github.com/deepseek-ai/DeepEP) 的低延迟内核输出。

#### 实用工具函数

除了上述内核外，本库还提供以下实用函数：

- `deep_gemm.set_num_sms`：设置最大可用的 SM 数量
- `deep_gemm.get_num_sms`：获取当前设定的最大 SM 数量
- `deep_gemm.get_m_alignment_for_contiguous_layout`：获取连续布局下分组级别的 M 对齐要求
- `deep_gemm.get_tma_aligned_size`：获取所需的 AIU 对齐尺寸
- `deep_gemm.get_col_major_tma_aligned_tensor`：获取列主序且 AIU 对齐的张量

本库还支持以下环境变量，可能对调试和调优有帮助：

- **通用设置**
  - `DG_JIT_DEBUG`: `0` 或 `1`，是否打印更多 JIT 调试信息，默认为 `0`
- **JIT 缓存相关**
  - `DG_JIT_CACHE_DIR`: 字符串，指定编译内核的缓存目录，默认为 `$HOME/.deep_gemm`
  - `DG_JIT_DISABLE_CACHE`: `0` 或 `1`，是否禁用缓存目录，默认为 `0`
- **HGCC/HGRTC 选择**
  - `DG_JIT_USE_NVRTC`: `0` 或 `1`，是否使用 HGRTC 替代 HGCC，可加快编译速度但某些情况可能影响性能，默认为 `0`
  - `DG_JIT_NVCC_COMPILER`: 字符串，指定编译器路径，默认从 `torch.utils.cpp_extension.CUDA_HOME` 中查找
- **编译器选项**
  - `DG_JIT_PTXAS_VERBOSE`: `0` 或 `1`，是否显示详细的 PTXAS 编译输出，默认为 `0`
  - `DG_JIT_PRINT_COMPILER_COMMAND`: `0` 或 `1`，是否打印 HGCC 编译命令，默认为 `0`
- **测试相关**
  - `DG_NSYS_PROFILING`: `0` 或 `1`，是否启用兼容 Asight Systems 的测试模式，默认为 `0`

更多示例和细节请参考 [测试代码](tests/test_core.py) 或查看相应的 Python 文档。

## 优化技术

我们用 🐳 标记那些未直接沿用自 CUTLASS 的创新技术。

#### 持久化 Warp 特化（Persistent Warp Interleave）

遵循 CUTLASS 的设计理念，DeepGEMM for PPU的内核采用 Warp Interleave设计，使得数据搬运、MMA 指令和 promotion操作能够重叠执行。

#### 常见细节优化

- 更大的块尺寸（最高达 256x256 🐳）

#### 统一且优化的块调度器

- 所有非分组和分组内核共用 [同一个调度器](deep_gemm/include/deep_gemm/scheduler.cuh)
- 采用 [光栅化（Rasterization）](https://github.com/NVIDIA/cutlass/blob/eefa171318b79cbe2e78514d4cce5cd0fe919d0c/media/docs/efficient_gemm.md#threadblock-rasterization) 策略以提升 L2 缓存复用率

#### 完全 JIT 化设计 🐳

DeepGEMM 采用完全的 [即时编译（JIT）](deep_gemm/jit) 设计，安装时无需编译。所有内核均在运行时通过轻量级 JIT 实现动态编译。这种方法带来诸多优势：

- 将 GEMM 形状、块大小和流水线级数作为编译时常量处理
  - 节省寄存器资源
  - 使编译器能进行更深层次优化
- 自动选择最优的块大小、warpgroup 数量、流水线级数
  - 无需自动调优，最优配置由确定性算法选出
- 完全展开 MMA 流水线，为编译器提供更多优化机会
  - 对小形状尤其重要
  - 详见 [内核文件](deep_gemm/include/deep_gemm/fp8_gemm.cuh) 中的 `launch_k_iterations` 函数

总体而言，JIT 显著提升了小形状矩阵的性能表现，其思路类似于 [Triton](https://github.com/triton-lang/triton/) 编译器。

#### 非对齐块大小 🐳

对于某些特定形状，若块大小强制对齐到 2 的幂次，可能导致 SM 利用率不足。例如，当 `M=256, N=7168` 时，若采用典型的 `BLOCK_M=128, BLOCK_N=128`，则仅有 `(256 / 128) * (7168 / 128) = 112` 个 SM 被利用（总共 132 个）。为此，我们支持非对齐块大小（如 112），使得 `(256 / 128) * (7168 / 112) = 128` 个 SM 可参与计算。将此技术与细粒度缩放结合需要精细优化，但最终能带来显著性能增益。

## 致谢

DeepGEMM 的设计灵感来源于 [CUTLASS](https://github.com/nvidia/cutlass) 项目。在此向所有开发者致以感谢与敬意！

## 许可证

本代码仓库依据 [MIT 许可证](LICENSE) 发布。

## 引用格式

```bibtex
@misc{deepgemm2025,
      title={DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling},
      author={Chenggang Zhao and Liang Zhao and Jiashi Li and Zhean Xu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepGEMM}},
}
```
