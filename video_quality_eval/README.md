# 视频质量-效率配对评测工具包

面向"视频扩散模型稀疏注意力加速"的评测，解决的问题是：**如何把"背景闪烁有没有变严重"变成可复现、可检验、能写进报告的数字。**

## 1. 指标矩阵：谁负责回答什么问题

| 指标 | 来源 | 方向 | 回答的问题 | 代价 |
|---|---|---|---|---|
| `bg_l1_mean/std/p95` | 本仓库（自研） | 越小越好 | 背景掩码内帧间像素残差有多大、是否突发 | 低 |
| `warp_resid_mean/std/p95` | 本仓库（自研） | 越小越好 | 背景变化能否被运动模型解释掉（不能解释 = 闪烁/漂移） | 中（光流） |
| `patch_dc_var_mean/median/p90` | 本仓库（自研） | 越小越好 | 背景是否"忽明忽暗、色块脉动"（最接近人眼感知） | 低 |
| `patch_dc_lowfreq_ratio` | 本仓库（自研） | 越小越好 | 背景脉动的能量有多少集中在低频（推导文档 P4 的直接检验量） | 低 |
| `lpips_bg_mean` | 本仓库封装 LPIPS | 越小越好 | 背景裁剪区域的感知差异（可选） | 中（需 GPU） |
| `flicker_index` | 本仓库（自研合成） | 越小越好（1.0 = 与 baseline 同级） | 上述三项相对配对 baseline 的几何平均 | 低 |
| `subject_consistency` | 官方 VBench（DINO 特征） | 越大越好 | 主体是否稳定 | 高 |
| `background_consistency` | 官方 VBench（CLIP 特征） | 越大越好 | 背景语义是否稳定 | 高 |
| `motion_smoothness` | 官方 VBench（帧插值） | 越大越好 | 运动是否平滑 | 高 |
| `temporal_flickering` | 官方 VBench（全画面帧间 MAD） | **需核对版本方向** | 全画面帧间差异（前景运动也会计入） | 中 |
| `dinov2/clip_consistency_proxy` | 本仓库（非官方） | 越大越好 | 调试期快速趋势信号，**不等于 VBench 数值** | 中 |
| `fvd` / `kid` | 本仓库封装 I3D/R3D | 越小越好 | 整体分布是否偏移（**对局部闪烁不敏感**） | 极高 |

关键设计：**VBench 的 `temporal_flickering` 是全画面 MAD，前景运动也会被计入**，所以它无法把"背景闪烁"和"正常运动"分开。本仓库的闪烁指标把计算限制在 **配对 baseline 提取出的低运动背景掩码** 上，二者之差才是"前景运动贡献"。这也是本工具包相对 VBench 的增量所在。

## 2. 配对：为什么所有结论都必须按 (prompt_id, seed) 配对

- 不同 prompt 的难度差异远大于方法差异，聚合平均会把信号淹没。
- 掩码必须来自 baseline（若来自被评测视频自身，"闪得越厉害 → 掩码越乱"是循环论证）。
- 配对 delta 才能用 Wilcoxon + bootstrap CI 给出"显著/不显著"的判定。

## 3. 两段式流程

1. **调试期**（几十次快速迭代）：`--metrics verify,flicker`（必要时加 `proxy`）。秒级到分钟级，用于看趋势、卡阈值。
2. **正式期**（写进报告的最终数字）：`--metrics verify,flicker,vbench,proxy` + 小规模 `fvd`。固定 prompt 集合、seed、步数。

```bash
# 调试期
python run_eval.py --manifest manifest.json --outdir results_dev --metrics verify,flicker

# 正式期（需要 VBench 与 vbench_full_info.json）
python run_eval.py --manifest manifest.json --outdir results_final \
  --metrics verify,flicker,vbench,proxy,fvd \
  --vbench-full-info /workspace/VBench/vbench_full_info.json \
  --vbench-dims subject_consistency,background_consistency,motion_smoothness,temporal_flickering \
  --timing timing.csv --n-boot 10000
```

## 4. manifest 格式

见 `manifest.example.json`。每个 run 至少需要：

```json
{"prompt_id":"p01","group":"high_motion","seed":123,"tag":"pisa_1st_0.8",
 "strategy":"order1_compensated","sparsity":0.8,"video":"/abs/path.mp4",
 "latency_s":182.0,"peak_mem_gb":16.8,"rho_mean":0.31}
```

- `tag`：方法配置（baseline 用 `dense`，或由 `baseline_tag` 指定）。
- `strategy`：用于画图分series（`dense` / `hard_drop` / `order1_compensated` / `hybrid`）。
- `rho_mean`：该配置下实测的"平均被丢弃注意力质量比例"，用来检验推导文档里的线性预测（可留空）。
- `started_ts`：实验开始的 Unix 时间戳，用于识别"复用了旧视频"的 cache 命中。

> 约定：`flicker_index` 相对配对 baseline 定义，baseline 行恒为 1.0（这是定义值，不是独立测量值，
> 报告中请说明）。`bg_l1_ratio` / `warp_resid_ratio` / `patch_dc_var_ratio` 同理。
> 若要比较两个非 baseline 方法，请分别与 baseline 比，不要直接比 ratio。

## 5. 完整性校验（先做的事）

`evaltools/videoutil.py` 会检查：文件存在、可解码、帧数/分辨率/fps 符合预期、修改时间不早于实验开始、以及**跨 run 的内容 sha256 重复**。

这一层专门防 Wan2.1 的两个坑：
- `cache_video failed: result type Float can't be cast to the desired output type Byte` —— 日志里打印了保存路径，但文件可能根本没写出来；
- cache-file 机制 —— 同名文件已存在时直接复用，得到的是**旧结果**。

## 6. 已知限制（请写进报告，不要藏起来）

1. 自研闪烁指标的绝对值没有物理量纲，只在**同分辨率、同 max_side、同一配对集合**内可比；报告时请写 `flicker_index`（相对 baseline）而不是原始值。
2. 光流有噪声地板，`warp_resid` 在低纹理背景上可能被高估。
3. 代理一致性不等于官方 VBench 数值，只用于趋势判断。
4. FVD 在 n<32 时置信区间极宽，只能当"没有崩"的证据，不能当"质量无损"的证据。
5. VBench 各版本的 API/CLI/结果 JSON 结构存在差异，`vbench_adapter.py` 优先做宽松解析，首次运行请核对日志中的 `mode=` 与 `results_json=`。

## 7. 自检

```bash
python selftest.py
```

会合成"静态背景 + 移动前景"与"同样内容 + 逐帧亮度抖动"两组视频，验证指标能把抖动组判为更差（`verdict=degraded`），并检查背景掩码比例是否合理。

自检产物在 `_selftest/`（合成数据，可随时删除）。图内文字使用 ASCII，因为服务器通常缺少中文字体，
中文会渲染成方框；需要中文标签时先装 Noto Sans CJK 并设置 `rcParams["font.sans-serif"]`。

## 8. 环境基线

```bash
python env_report.py --outdir results/env
```

采集 GPU 型号/显存/时钟/降频状态、torch/CUDA、注意力后端（flash_attn / FA3 / triton / FlexAttention 是否可用）、
CPU/内存/磁盘、关键依赖版本，产出 `env_report.md` + `env_report.json`。
把它附在报告的"实验设置"一节——加速比结论必须绑定一个明确的后端与硬件口径，否则不可复现。
