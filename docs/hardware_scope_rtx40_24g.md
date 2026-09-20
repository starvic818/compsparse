# 硬件画像与实验范围裁剪（Compshare 单卡 RTX40 系）

依据你给的实例信息：`RTX40系 ×1`、14 核、64GB 内存、50GB 系统盘（已用 39.89% ≈ 20GB）、
镜像 `cuda128_torch291_py312`、单卡。

> 先用 `nvidia-smi` 确认三件事：型号（4090 / 4090D，24GB；若实为 4080 则只有 16GB）、
> `compute_cap`（应为 8.9 = Ada）、以及 `clocks.sm / temperature / power.draw`（散热降频会影响计时）。
> 下面按 **24GB Ada** 规划，并在文末给出若为 16GB 时的额外裁剪。

---

## 1. 三条立即生效的结论

**① 14B 模型出局，实验主线锁定 Wan2.1-T2V-1.3B。**
14B 以 bf16 存储需要约 28GB 权重，超过 24GB；唯一的替代路线是量化 + CPU offload，
但那会让"加速比"的口径被 offload 与量化双重污染，结论不可比。
因此"超长序列"这条主线要用**序列长度**（帧数、分辨率）来支撑，而不是用模型规模支撑。

**② FlashAttention-3 在这台机器上不可用。**
FA3（`flash_attn_interface`）是 Hopper（sm90）专属，Ada（sm89）上跑不了。
所以"基于 Flash Attention 底层算子"的现实落点是 **FA2 / PyTorch SDPA / FlexAttention(Triton)**。
这直接决定报告里必须写清一件事：**你的"标准 Flash Attention 基线"具体是哪一个后端**。
Wan 的 `wan/modules/attention.py` 会在缺少 `flash_attn` 时静默回落到 SDPA——
如果你不知道当前基线走的是哪条分支，加速比就没有可比性。

**③ 50GB 系统盘是隐性瓶颈，先算账再装东西。**

```text
T5 umt5-xxl 文本编码器 (bf16)   ~ 11 GB      <- 大头
DiT 1.3B (bf16)                 ~ 2.7 GB
VAE                             ~ 0.5 GB
输出视频 (100 段 × 约 3MB)       ~ 0.3 GB
VBench 依赖 + 预训练权重         ~ 3-6 GB    <- AMT/MUSIQ/CLIP/DINO
pip/conda 缓存                   ~ 1-3 GB
合计                             ~ 18-24 GB  （当前剩余约 30GB，够但不宽裕）
```

动 VBench 之前先 `df -h` 和 `pip cache purge`；不要把 fp16 / fp8 / bf16 三份权重同时留在盘上。

---

## 2. 显存预算：真正的坑是"注意力矩阵"，不是权重

权重部分很轻松：DiT 2.7GB + VAE 0.5GB，T5 放 CPU（`--t5_cpu`）。麻烦在激活。

| 配置 | N (token) | 单个 head 的 score 矩阵 (batch=1) | 含 CFG (batch=2) | 12 head 全物化 |
|---|---|---|---|---|
| 480p / 81f | 32760 | 2.15 GB | 4.3 GB | **51 GB → 必 OOM** |
| 480p / 161f | 63960 | 8.2 GB | 16.4 GB | 远超显存 |
| 720p / 81f | 75600 | 11.4 GB | 22.9 GB | 远超显存 |

FFN 中间激活（N × 8960 × 2B，含 CFG）：480p 约 2.4GB，720p 约 5.4GB——这些还在可控范围。

**结论**：任何"先把 attention 矩阵算出来、再套 mask"的原型写法，在 480p 就会爆显存。
原型阶段必须二选一：

```text
方案 A（推荐）: torch.nn.attention.flex_attention，用 block-sparse mask modifier
                —— torch 2.9 自带、走 Triton，Ada 上可用，而且直接给你真实的 kernel 级收益
方案 B（仅用于正确性对照）: 按 head 循环 + 分块 query 的参考实现，且只在 N≈4096 的小序列上验证
陷阱提醒: 不要生成 mask 的副本再 masked_fill；用就地置 -inf，否则显存再翻一倍
```

---

## 3. 时间预算：用你现有的 5.2 s/it 反推

```text
每步 FLOPs（480p/81f，含 CFG 两条分支）≈ 5.4e14
实测 5.22 s/step  ->  等效约 1.0e14 FLOP/s
RTX 4090 的 bf16 稠密算力量级在 10^2 ~ 3x10^2 TFLOPS（取决于累加精度），
实测 GEMM 通常落在 1.5e14 上下。
=> 你目前大致在 30%~70% 的区间，说明还有余量，但"关掉 offload 就能翻倍"是不现实的。
```

值得做的两件事：

1. **关掉 `--offload_model`**（改成 `--offload_model False --t5_cpu True`）。DiT 只有 2.7GB，
   24GB 卡上完全不需要 offload；offload 让权重每步走 PCIe，会持续稀释 kernel 层收益。
   预期拿回几个百分点到约 20%，**务必实测并写进报告**（这本身就是一条小结论）。
2. **一次加载跑完所有配置**。你现在每条命令 14 分钟里有约 2 分钟是加载 T5/模型；
   一轮 64 条配置就是白烧 2 小时。写一个循环脚本，模型只加载一次。

另外：VAE 解码与文本编码**不要算进加速比**。分段计时（T5 / 采样循环 / VAE），只对采样循环报加速比。

---

## 4. 内核路线（Ada 上的可行性排序）

```text
1. FlexAttention + block-sparse mask           最推荐：零编译成本、真实 kernel 级加速、可解释
2. 官方 PISA/稀疏 kernel（若为 FA2 CUDA 实现）  需编译：14 核下单卡编译约 30-60 分钟，
                                               设 MAX_JOBS=8 防 OOM；先看有没有现成 wheel
3. 自己写 Triton                                只有 1/2 都走不通时再考虑
4. FA3                                          不要尝试（Hopper 专属）
```

先跑 `python -c "import flash_attn; print(flash_attn.__version__)"` 确认镜像里到底有没有 FA2，
以及 Wan 当前走的是哪条分支。这一步决定了你后续所有"加速比"的分母是什么。

---

## 5. 计时卫生（单卡实例容易被忽略的四点）

```text
① 显式锁定 SDPA 后端: torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION])
   否则 PyTorch 会按形状自动切换后端，你测到的"加速"可能来自后端切换而不是你的算法。
② 记录 clocks.sm / temperature / power.draw（env_report.py 已采集），
   Compshare 实例散热与降频会造成 5%-10% 抖动，报告里附上这些数字比"我测了三次"更有说服力。
③ 每配置 ≥3 次重复，取中位数并报告标准差；第一次运行（warmup）永远不算。
④ 注意定时关机时间：长扫描前预估总时长，被中途杀掉 = 前面全部作废。
   无卡模式启动适合写代码/装包，不要用它跑实验。
```

---

## 6. 建议的实验矩阵裁剪（24GB 版）

| 配置 | N | 建议 | 理由 |
|---|---|---|---|
| 480p / 81f | 32760 | ✅ 主线 | 你已验证可跑，作为对照底座 |
| 480p / 161f | 63960 | ✅ 长序列主证据 | 显存比 720p 友好得多，长度接近 |
| 720p / 81f | 75600 | ⚠️ 仅作单点确认 | 需要 memory-efficient attention；FFN 激活含 CFG 约 5.4GB |
| 1080p / 81f | >170k | ❌ 放弃 | N² 与激活都会 OOM |
| 14B 的任何配置 | — | ❌ 放弃 | 权重 28GB > 24GB |

**用 480p/161f 替代 720p/81f 作为"超长序列"的主证据，是这台机器上更划算的选择**（64k vs 76k token，量级相同但显存压力低一档）。
这样你依然能画出"加速比随序列长度上升"的核心曲线，而且不会把几天时间烧在 OOM 调试上。

若 `nvidia-smi` 显示实际是 16GB（4080 级）：额外放弃 720p/81f，长序列轴改用 480p/121f 与 480p/161f。

---

## 7. 现在就该跑的命令

```bash
nvidia-smi
python env_report.py --outdir results/env        # 产出环境基线，附在报告附录
df -h && pip cache purge
python -c "import flash_attn, torch; print(flash_attn.__version__, torch.__version__)"
ls -lh Wan2.1/Wan2.1-T2V-1.3B                    # 核对权重体积，确认磁盘预算
```
