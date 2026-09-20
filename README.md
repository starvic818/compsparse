# compsparse

**Compensated sparse attention for long-video diffusion transformers.**

在视频 DiT 的全时自注意力中，用「精确计算 + 一阶泰勒补偿 + 尾部丢弃」的三段式策略，
替代传统的「保留 / 丢弃」二分稀疏，并提供一整套**先测量、后决定**的工具：
可稀疏性分析、误差界与开销公式、配对质量评测（含背景闪烁）、批量实验脚本。

> 状态：离线部分（数学推导、参考实现、可行性分析、评测套件、执行脚本）已完成并通过测试；
> 端到端实验等待 GPU 资源。**所有结论都标注了验证方式，未验证的会明确写"待卡"。**

---

## 为什么不是"又一个 top-k 稀疏"

这个项目最反直觉的结论是：**稀疏率不是自由参数，而是由注意力 logits 分布决定的预算。**

```text
一阶泰勒近似的单 key 相对误差 = 1 - e^{-δ}(1+δ)，δ = logit - 行最大值 ≤ 0
    → break-even 恰在 |δ| = 1：|δ|<1 时补偿优于丢弃，|δ|>1 时 1+δ 变号，补偿反而更差
    → 所以必须分三段：近端补偿、中段精确、远端丢弃

补偿开销 / 稠密注意力 ≈ (1+ρ)d / (2N)
    → 480p/81f (N=32760) 时仅 0.35%；折合端到端：纯丢弃 2.40x，一阶补偿 2.39x
    → 也就是说，补偿几乎是免费的，但只在正确的区间里才有效

补偿的有效区间（合成数据标定）：logit std ≲ 0.35
    std = 0.056 时补偿把误差压掉 373 倍；std = 0.674 时收益为 1.0（A 段为空，补偿退化为丢弃）
```

由此还有一条可证伪的推论：既然一阶近似在 δ≈0 处是**精确**的，
那么注意力质量最大的 key 恰恰是补偿最准的地方 —— **把精确预算全花在最高 logit 上可能是浪费的**。
合成数据上「中段精确」在 std≈0.2–0.4 区间误差低 2–4 倍（待真实数据复核）。

---

## 目录结构

```text
compsparse/
├── sparse_attn_proto/          方法原型与可行性分析（纯 CPU 可跑）
│   ├── sparse_attn.py          三段式实现：一阶矩 M_A/S_A/V_A、二阶矩 G_A
│   ├── test_sparse_attn.py     11 项单元测试（代数正确性 + 误差界 + 区间行为）
│   ├── feasibility.py          可稀疏性分析：相图、分区对照、误差代理标定
│   └── example_output/         样例输出（相图、对照表）
├── video_quality_eval/         配对质量评测
│   ├── run_eval.py             一键评测：校验 → 指标 → 配对统计 → 报告与图
│   ├── evaltools/flicker.py    背景闪烁指标（帧间 L1 / 光流残差 / 分块 DC / 低频能量占比）
│   ├── evaltools/vbench_adapter.py  官方 VBench 三件套适配
│   ├── evaltools/fvd.py        FVD / KID（含 bootstrap CI 与样本量警告）
│   ├── evaltools/paired.py     配对统计（bootstrap CI、Wilcoxon、效应量、达标判定）
│   ├── env_report.py           环境基线采集（GPU/后端/版本）
│   └── selftest.py             合成数据自检（抖动组必须被判为 quality degraded）
├── run_scripts/                实验执行
│   ├── make_experiment.py      prompt 集 / sweep 配置 / manifest 生成
│   ├── capture_qkv.py          抓真实 q/k/v，直接算出「每层每头能稀疏到多少」
│   └── sweep.py                批量生成：一次加载、断点续跑、CUDA events 计时
├── docs/
│   ├── pisa_exact_plus_approx_derivation.md   精确项+近似项推导（三段判据、误差界、开销公式）
│   ├── research_plan_v2.md                    任务书与进度表（含 D1–D5 决策点）
│   ├── hardware_scope_rtx40_24g.md            单卡 24GB 的实验矩阵裁剪
│   ├── offline_todo.md                        无卡期任务清单与上卡 runbook
│   └── REPO_SETUP.md                          建仓说明（描述 / 许可 / 可见度）
└── LICENSE                     Apache-2.0（用 GitHub 模板或 curl 生成，见文末）
```

---

## 快速开始（无需 GPU）

```bash
# 1) 代数正确性与误差界：11 项测试
python sparse_attn_proto/test_sparse_attn.py

# 2) 误差代理标定：确认代理在补偿主导区间偏差约 ±30%
python sparse_attn_proto/feasibility.py --calibrate --outdir results_cal

# 3) 相图：误差 vs 稀疏率，多条 logits 散布曲线
python sparse_attn_proto/feasibility.py --synthetic --compare --outdir results_feasibility

# 4) 评测套件自检（合成"静态背景 + 亮度抖动"视频组）
python video_quality_eval/selftest.py
```

有 GPU 后（单卡 24GB 即可，模型为 Wan2.1-T2V-1.3B）：

```bash
python run_scripts/make_experiment.py --outdir manifests --video-dir ./works
python run_scripts/capture_qkv.py --max-sample-steps 1 --layers 0 --heads 0 --outdir captures
python run_scripts/sweep.py --config manifests/sweep_config.json --outdir results --runner inproc
python video_quality_eval/run_eval.py --manifest results/manifest.json \
    --outdir results/eval --metrics verify,flicker
```

---

## 评测设计（为什么不用 FVD 当主指标）

“背景闪烁有没有变严重”是本项目的核心问题，而 **FVD 是分布距离，对局部时序不稳不敏感**。
因此主指标换成**配对比较**：

| 指标 | 方向 | 说明 |
|---|---|---|
| `flicker_index` | 越小越好 | 三项背景指标相对配对 baseline 的几何平均（1.0 = 同级） |
| `bg_l1_mean` | 越小越好 | 背景掩码内帧间像素差的均值/方差/p95 |
| `warp_resid_mean` | 越小越好 | 光流把 t+1 对齐回 t 后的残差（背景漂移/非刚性变化） |
| `patch_dc_var_mean` | 越小越好 | 低通后分块 DC 的时间方差（人眼最典型的"忽明忽暗"） |
| `patch_dc_lowfreq_ratio` | 越小越好 | 背景脉动能量中低频占比（检验"补偿把低频抖动转为高频噪声"） |
| VBench `background/subject consistency`、`motion_smoothness` | 越大越好 | 官方实现，定稿数字使用 |
| FVD / KID | 越小越好 | **仅作补充**，样本量 < 32 时只报趋势 |

两个关键设计：**背景掩码来自配对的 baseline 视频**（否则"闪得越厉害→掩码越乱"是循环论证）；
所有结论按 `(prompt_id, seed)` 配对后做 bootstrap 置信区间与 Wilcoxon 检验。

---

## 硬件边界

在单卡 24GB 上：模型锁定 **Wan2.1-T2V-1.3B**（14B 的 bf16 权重 28GB > 24GB）；
FlashAttention-3 是 Hopper 专属，Ada（RTX40 系）需用 FA2 / SDPA / FlexAttention；
**长序列主轴用 480p/161f（N≈64k）而不是 720p/81f**，显存压力低一档、token 量级相当。

还有一个必须避开的坑：不要把 attention 矩阵物化出来再套 mask ——
480p 时单个 head 的 score 矩阵就是 2.15GB（含 CFG 4.3GB），12 个头必 OOM。

---

## 进度

```text
✅ 精确项 + 近似项推导（三段判据、误差界、开销公式）
✅ 三段式参考实现 + 11 项单元测试
✅ 可稀疏性分析（相图 / 分区对照 / 误差代理标定）
✅ 配对质量评测套件（闪烁指标 + VBench 适配 + FVD + 统计报告）
✅ 实验执行脚本（配置生成 / 抓取 / 批量生成）+ 无卡 dry-run
⬜ 真实 logits 分布测量（决定补偿是否有效用之地）
⬜ 层内微基准、端到端加速比、配对质量结论
```

---

## 与 PISA 的关系

本项目受 PISA 式 piecewise 稀疏策略启发，但**是独立实现与分析**：
推导部分给出了显式的截断误差上界、补偿开销的闭式表达、以及"精确预算该花在哪"的对照实验，
并明确记录了与原文可能存在的机制差异（见 `docs/` 中的核对清单）。
使用前请自行核对原论文与官方实现。

## 致谢

- [Wan2.1](https://github.com/Wan-Video/Wan2.1)（Apache-2.0）：被加速的视频生成基座模型
- [VBench](https://github.com/Vchitect/VBench)：官方质量维度评测
- FlashAttention / PyTorch SDPA / `torch.nn.attention.flex_attention`：注意力后端

## License

Apache-2.0（见 `LICENSE`）。许可证文本请用 GitHub 的模板或官方源生成，不要手抄：

```bash
curl -o LICENSE https://www.apache.org/licenses/LICENSE-2.0.txt
```
