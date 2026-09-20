# 实验执行脚本（无卡期即可写完并验证）

三个脚本覆盖"配置 → 抓取 → 批量生成 → 评测"的完整链路。**全部支持无卡运行**：
`--dry-run` 不加载模型、不需要装 `wan`，只用来看计划是否正确。

```text
make_experiment.py   生成 prompt 集（16 条 / 4 类）、sweep 配置、manifest 模板
       ↓
capture_qkv.py       上卡第一天：抓真实 q/k/v，直接算出"这层这个头能稀疏到多少"  ← 决策点
       ↓
sweep.py             批量生成：一次加载跑完、断点续跑、记录 latency/peak_mem、回填 manifest
       ↓
video_quality_eval/run_eval.py    配对质量评测（闪烁指标 + VBench + FVD）
```

## 1. 生成实验配置

```bash
python make_experiment.py --outdir manifests --video-dir /workspace/Wan2.1/works
```

产出 `prompts.json`（4 类 × 4 条，含你已在用的 3 条）、`sweep_config.json`、`manifest_template.json`。
按 5.2 s/it × 50 步估算，一轮 96 条配置 ≈ **7.4 GPU-小时**（480p，不含 720p）。

## 2. 抓真实注意力（上卡第一天，决策点）

```bash
# 无卡验证计划
python capture_qkv.py --dry-run --layers 0,15,29 --heads 0,5,11 --steps 0,25
# 有卡（先最小验证，再扩大）
python capture_qkv.py --max-sample-steps 1 --layers 0 --heads 0 --outdir captures
```

480p/81f 时 N≈32760，抓 3 层 × 3 头 × 2 步 ≈ 18 组、约 288MB（fp16）。
产出 `capture_summary.csv`（每层每头一行）与 `capture_report.md`（含结论与下一步建议）。

**这一步会回答整个项目最关键的问题**：真实 logits 的 std 落在哪个区间。
std ≲ 0.35 → 补偿有用武之地；std ≳ 0.35 → A 段为空、补偿退化成丢弃，
此时应把重心转向"按头选择稀疏策略"的边界分析（同样是有价值的结论）。

## 3. 批量生成

```bash
# 无卡验证
python sweep.py --config manifests/sweep_config.json --outdir results_sweep --dry-run --limit 3
# 有卡：先 smoke test 两条，确认保存与计时都对
python sweep.py --config manifests/sweep_config.json --outdir results_sweep --runner inproc --limit 2
# 正式跑（断点续跑，中断后重跑同一条命令即可）
python sweep.py --config manifests/sweep_config.json --outdir results_sweep --runner inproc
```

关键特性：

```text
[x] 一次加载跑完所有配置（省掉每条 2 分钟的加载时间）
[x] 断点续跑：输出已存在且帧数正确就跳过
[x] 计时用 CUDA events，只统计采样循环；peak_mem 用 max_memory_allocated
[x] save_video_robust()：显式 uint8 转换，绕开原版 cache_video 的 dtype 报错
[x] 自动回填 manifest.json，直接喂给 run_eval.py
[x] 稀疏方法接入：在 sweep_config.json 的 tag 里写 "patch": "my_patch:install"
```

## 4. 评测

```bash
python video_quality_eval/run_eval.py --manifest results_sweep/manifest.json \
    --outdir results_sweep/eval --metrics verify,flicker
```

## 无卡期的推荐顺序

```text
1. make_experiment.py            生成配置（1 分钟）
2. sweep.py --dry-run            检查命令拼装、路径、run 数（1 分钟）
3. capture_qkv.py --dry-run      检查抓取计划、估算落盘体积（1 分钟）
4. CPU 跑一次 smoke 生成（9 帧/256p/3 步），验证保存与替换点（见 `docs/offline_todo.md` 第 3 节）
5. 填完 `docs/offline_todo.md` 第 1 节的 PISA 机制核对清单
```
