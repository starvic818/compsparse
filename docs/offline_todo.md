# 无卡期任务清单（按"决定上卡后能否一次跑通"排序）

原则：**无卡期只做两类事** ——（a）决定实验设计是否成立的事（读原文、验代数、算可行性）；
（b）消掉上卡后必然踩的坑（保存失败、参数没锁、脚本要重跑）。
不要在无卡期做的事：跑完整推理（CPU 上 1.3B 生成 81 帧要几小时，得到的耗时数据也没有意义）。

Compshare 的"无卡模式启动"就是为这类工作准备的（不占卡、便宜），装包、写代码、跑 CPU 测试都在这个模式下做。

---

## 优先级总表

| # | 任务 | 为什么现在做 | 产出 | 预计 |
|---|---|---|---|---|
| 1 | PISA 原文机制核对 | 若"补偿项"与方案假设不符，后面全部实验设计要重写 | `pisa_mechanism.md` | 2–4h |
| 2 | 跑通 CPU 参考实现与相图 | 代数正确性 + "能稀疏到多少"无需 GPU 就能答 | 已交付，复核并记录 | 1h |
| 3 | CPU 上做"管线连通性测试" | 提前暴露保存失败、设备选择、hook 位置等问题 | 可运行的改动 patch | 2–4h |
| 4 | 修 `cache_video` 保存问题 | 现在三组 baseline 视频很可能没落盘 | 可播放的 mp4 | 1–2h |
| 5 | 锁定参数 + 生成 manifest | 加速比与质量结论都依赖配对 | `manifest.json` | 1h |
| 6 | 写 sweep runner（一次加载跑完） | 上卡后每条命令白烧 2 分钟加载 | `sweep.py` | 2–3h |
| 7 | 报告骨架 + 图表模板 | 写作是最容易拖到最后的环节 | `report_draft.md` | 2h |
| 8 | 无卡模式装环境、清磁盘 | 上卡后时间很贵，装包不该占卡时 | 依赖齐备 | 1h |

---

## 1. PISA 原文机制核对（最高优先）

必须回答这 8 个问题，答案写进 `pisa_mechanism.md`（能引用原文/代码行号最好）：

```text
Q1 泰勒展开的变量是 logit s_ij，还是 softmax 概率 p_ij？
Q2 补偿项到底是什么形式？是"把被丢 key 的贡献近似加回"，还是只用于"挑块的打分"？
   （如果是后者，就不存在可消融的"补偿项"，方案里的消融实验要整体重写）
Q3 "hybrid order" 具体指什么？是不是"分子一阶 + 分母二阶"？
Q4 精确集/近似集的划分规则：按 logit 幅值、按块均值，还是按位置窗口？
Q5 是否有 sink / 首帧 / 文本 token 的强制精确参数？
Q6 是否按 head / layer 设置不同稀疏率？
Q7 原文的加速比是在多大序列长度上测的？（补偿开销 ∝ 1/N，不能直接搬到 480p）
Q8 官方实现依赖哪个注意力后端（FA2 / Triton / 自写 CUDA）？Ada 架构能不能跑？
```

**完成定义**：能明确回答"这个方法的补偿项是否真实存在、在什么条件下成立、能否用于视频 DiT"。
如果 Q2 的答案是"只用于挑块"，那就把项目重心从"复现补偿"改为
**"测量不同稀疏选择策略在视频 DiT 上的质量-效率边界"**，仍然是一份合格的实践报告，
而且你手里已有的 `feasibility.py` 正好是这个新方向的工具。

---

## 2. 复核 CPU 参考实现与相图

已经交付（`sparse_attn_proto/`，相对仓库根目录），无卡期只做三件事：

```bash
cd sparse_attn_proto
python test_sparse_attn.py                    # 11 项测试，确认 ALL PASS
python feasibility.py --synthetic --outdir results_feasibility --n 256 --d 64
# 换更接近 Wan 的设置：head_dim=128，但把 N 控制在 512 以内（CPU 可承受）
python feasibility.py --synthetic --outdir results_feasibility_d128 --n 384 --d 128
```

**完成定义**：得到你自己的相图（误差 vs 稀疏率，多条 logit 散布曲线），
并把"补偿有效区间"的边界数值写进报告方法章节的草稿。

注意 `--d 128` 时 `M_A`/`G_A` 是 128×128 的逐行张量，CPU 上 N 别超过 ~512，否则内存会吃紧。

---

## 3. CPU 上做"管线连通性测试"（性价比最高的无卡任务）

目标不是测速度，而是验证**改动后整条链路不崩**。做法：小规模配置跑 2–4 步采样。

```bash
# 9 帧 + 小分辨率 + 极少步数：N≈768 token，CPU 上单步只有几十秒
python generate.py --task t2v-1.3B --size 256*256 --frame_num 9 \
  --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu True \
  --sample_steps 3 --base_seed 12345 \
  --prompt "a static red cube on a table" --save_file ./works/smoke_cpu.mp4
```

验收清单：

```text
[ ] 能跑完不崩（CPU 上无 CUDA 时设备选择是否正确？不行就手动把 device 改成 "cpu"）
[ ] mp4 真的写出且能播放（这一步会暴露 cache_video 的 dtype 问题）
[ ] 在 wan/modules/model.py 的 WanSelfAttention.forward 里插入 print/log，
    确认你的替换点在 CFG 两条分支上都被调用到
[ ] hook 能抓到 q/k/v 的形状（预期 [B, H, N, D] 或 [B, N, H, D]，记录下来）
[ ] 记录 N 的实测值：256p/9f 时 N=?，用来校准你的 N = f(分辨率, 帧数) 公式
```

这一步能把"上卡后第一天就卡在保存失败/形状不对"的风险直接消掉。

---

## 4. 修 cache_video 保存问题

你日志里的 `cache_video failed, error: result type Float can't be cast to the desired output type Byte`
意味着那几次带 `--save_file` 的运行很可能没有真正写盘。无卡期就能定位并修掉：

```bash
python -c "import numpy, imageio; print(numpy.__version__, imageio.__version__)"
# 在 Wan2.1 目录下找到 cache_video / save_video 的实现（通常在 wan/utils/utils.py）
grep -rn "cache_video\|def save_video" Wan2.1/wan/utils/ | head
```

常见根因是新版 numpy/imageio 在 `(x * 255).astype(np.uint8)` 或 `np.clip(..., out=uint8)` 这类
就地写回时类型不匹配。最小改法：在写盘前显式做 `arr = np.clip(arr, 0, 255).astype(np.uint8)`，
或直接把帧序列写成 PNG + ffmpeg 封装，绕开这条转换链。

**完成定义**：用一段随机张量（不需要模型）走一遍保存函数，能稳定产出可播放的 mp4。

---

## 5. 锁定参数 + 生成 manifest

```text
固定项: seed / sample_steps / solver / shift / CFG / offload / 分辨率 / 帧数 / 模型 commit
prompt 集: 16 条，覆盖 high_motion / low_motion / complex_texture / static_background 四类
           （static_background 那类专门用来测 P4：背景频率成分转移）
```

用 `video_quality_eval/manifest.example.json` 的格式，
基于你已有的三组 prompt（cyberpunk / water droplet / forest）扩到 16 条，
每条 prompt 固定 2 个 seed。**关键：baseline 与稀疏方法必须共享同一 seed**，
所以生成命令里加 `--base_seed <固定值>`（Wan 的 `generate.py` 支持该参数，日志里能看到）。

---

## 6. 写 sweep runner（上卡后的省时主力）

要求：

```text
[ ] 模型只加载一次，循环跑 (策略 × 稀疏率 × prompt × seed)
[ ] 断点续跑：输出文件已存在且校验通过就跳过（用 sha256 或帧数校验）
[ ] 每条记录追加写入 timing.csv：prompt_id,tag,sparsity,latency_s,peak_mem_gb,steps,commit
[ ] latency 只统计采样循环（CUDA events），排除模型加载与文本编码
[ ] 自动生成/更新 manifest.json 交给评测工具
```

无卡期可以先写出来，用 `--dry-run` 只打印将要执行的命令序列来验证参数拼装是否正确。

---

## 7. 报告骨架

先把结构和小节标题写好，上卡后只填数字：

```text
1 背景与问题：视频 DiT 的注意力计算墙（用你实测的 N 与 FLOPs 占比）
2 方法：精确项 + 近似项拆分；三段划分；误差上界；补偿开销 (1+ρ)d/(2N)
3 实现：接入 Wan2.1 的位置、sink/首帧处理、CFG、后端选择（FA2/SDPA/FlexAttention）
4 实验：环境基线 / 层内微基准（V1）/ ρ 分布（V2）/ 配对质量-效率（V3）/ 频率成分（V4）
5 结论与局限：包括证伪路径
附录：env_report.md、manifest、复现命令
```

加上一张 related work 对照表：Sparse VideoGen / MInference / SpargeAttn / Sparse-vDiT，
每行写"它的稀疏粒度 / 是否补偿 / 用在什么模型 / 与你的差异"。

---

## 8. 无卡模式装环境与清磁盘

```bash
df -h && pip cache purge
pip install -r video_quality_eval/requirements.txt
git clone https://github.com/Vchitect/VBench.git && pip install -e VBench   # 装完先别跑，等有卡
python video_quality_eval/env_report.py --outdir results/env          # 无卡也能采到软件版本
```

注意 `env_report.py` 在没有 GPU 时会把 GPU 段标成 `available: False`，
上卡后再跑一次才是完整的基线。

---

## 上卡第一天的执行顺序（Runbook）

```text
0. nvidia-smi && python video_quality_eval/env_report.py --outdir results/env
1. CPU 管线连通性测试的配置改成 480p/81f，跑 1 条 dense baseline，确认耗时与显存
2. 打开 hook，抓 3 个层（0 / 15 / 29）的 q/k/v，导出 qkv.pt 与 delta.npy
3. python feasibility.py --delta-file delta.npy --qkv-pt qkv.pt
   -> 此时你才第一次知道：真实模型里补偿到底有没有用武之地（这是本周最重要的一个数字）
4. 根据第 3 步结果决定：继续做补偿，还是转做"稀疏选择策略的边界分析"
5. 再开始 sweep（一次加载跑完，断点续跑）
```

第 3 步是整条链路的决策点：如果真实 logits 的 std 落在 0.5 以上，
就说明 A 段为空、补偿无用，越早发现越好——**这个结论本身就值一篇报告的一节**。
