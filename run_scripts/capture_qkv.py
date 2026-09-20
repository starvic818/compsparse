"""抓取 Wan2.1 真实注意力张量，并直接给出"这层这个头能稀疏到多少"。

这是上卡第一天的决策点：**真实模型里泰勒补偿到底有没有用武之地**。
方法：包装 Wan 的 attention 调用，在指定 (层, 头, 采样步) 上抓 q/k/v，
计算 delta = (q·k)*scale - rowmax，然后用只依赖 delta 的 delta_profile 给出可稀疏性画像。

输出：
    qkv_capture.pt          被抓的 q(子采样行)/k/v/delta（供 feasibility.py 做精确误差分析）
    capture_summary.csv     每 (层, 头, 步) 一行的统计（logit std / 熵 / 顶 1% 质量占比 /
                            各稀疏率下的 rho 与误差代理 / 最大安全稀疏率）—— 直接进报告
    capture_report.md       人读摘要 + 下一步建议

用法：
    # 无卡也能验证计划是否正确
    python capture_qkv.py --dry-run --layers 0,15,29 --heads 0,5,11 --steps 0,40

    # 有卡时真正抓取（只跑少量采样步，不生成完整视频）
    python capture_qkv.py --task t2v-1.3B --ckpt_dir ./Wan2.1-T2V-1.3B --size 832*480 \
        --layers 0,15,29 --heads 0,5,11 --steps 0,25 --max-queries 192 \
        --prompt "a static red cube on a table" --outdir captures

注意：
  · 只跑少量采样步（例如只跑前 26 步就退出），不要生成完整视频。
  · 抓取会引入同步开销，所以不要在测速运行里开启。
  · 不同 Wan 版本的模块路径可能不同，脚本会自动探测并打印找到了什么。
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------------------------------------------ 模块探测

def find_attention_callable() -> Tuple[Any, str]:
    """返回 (模块对象, 属性名)，该属性是 Wan 用来做 self-attention 的函数。"""
    candidates = [
        ("wan.modules.model", "flash_attention"),
        ("wan.modules.attention", "flash_attention"),
        ("wan.modules.model", "attention"),
        ("wan.modules.attention", "attention"),
    ]
    for mod_name, attr in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, attr) and callable(getattr(mod, attr)):
            return mod, attr
    raise RuntimeError(
        "找不到 Wan 的 attention 函数。已尝试：%s\n"
        "请打开 wan/modules/model.py，看 self-attention 里调用的函数名与来源，"
        "然后把它加到本脚本的 candidates 列表里。"
        % ", ".join("%s.%s" % c for c in candidates)
    )


def to_bhld(x, num_heads: int):
    """把 Wan 的 q/k/v 统一成 [B, H, L, D]。
    兼容两种常见布局：[B, L, H*D]（3 维，需要 reshape+transpose）与 [B, H, L, D]（4 维）。"""
    import torch

    if not isinstance(x, torch.Tensor):
        return None
    if x.dim() == 3:
        b, l, hd = x.shape
        if num_heads <= 0 or hd % num_heads != 0:
            return None
        return x.view(b, l, num_heads, hd // num_heads).transpose(1, 2)
    if x.dim() == 4:
        a, b2, c, d = x.shape
        if a == num_heads:                      # [H, L, B, D]
            return x.permute(2, 0, 1, 3)
        if b2 == num_heads:                     # [B, H, L, D]
            return x
        if c == num_heads:                      # [B, L, H, D]
            return x.transpose(1, 2)
    return None


# ------------------------------------------------------------------ 统计

def delta_stats(delta, sparsities, tau: float, target_err: float) -> Dict[str, Any]:
    """对 delta [rows, N] 计算统计量与可稀疏性画像。"""
    import torch

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "sparse_attn_proto"))
    from feasibility import delta_profile, max_safe_sparsity  # type: ignore

    w = torch.exp(delta)
    p = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
    n = delta.shape[-1]
    top1pct = max(1, n // 100)
    top_mass = torch.topk(p, top1pct, dim=-1).values.sum(dim=-1)

    out: Dict[str, Any] = {
        "n_keys": int(n),
        "n_queries": int(delta.shape[0]),
        "logit_std": float(delta.std()),
        "delta_p10": float(delta.quantile(0.10)),
        "delta_p50": float(delta.quantile(0.50)),
        "delta_p90": float(delta.quantile(0.90)),
        "attn_entropy": float(entropy.mean()),
        "attn_entropy_norm": float((entropy / math.log(n)).mean()),
        "top1pct_mass": float(top_mass.mean()),
    }
    prof = delta_profile(delta, sparsities, tau=tau, target_err=target_err)
    for r in prof:
        tag = "s%02d" % int(round(r["sparsity"] * 100))
        out["rho_drop_" + tag] = r["rho_drop_median"]
        out["rho_comp_" + tag] = r["rho_comp_median"]
        out["proxy_err_" + tag] = r["proxy_err_median"]
        out["band_keys_" + tag] = r["band_keys_median"]
    safe = max_safe_sparsity(delta, target_err=target_err, tau=tau)
    out["max_safe_sparsity"] = safe["max_safe_sparsity"]
    out["optimistic_flag"] = bool(prof[-1]["optimistic"]) if prof else False
    return out


# ------------------------------------------------------------------ 主流程

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="不加载模型，只打印抓取计划")
    ap.add_argument("--task", default="t2v-1.3B")
    ap.add_argument("--ckpt_dir", default="./Wan2.1-T2V-1.3B")
    ap.add_argument("--size", default="832*480")
    ap.add_argument("--frame_num", type=int, default=81)
    ap.add_argument("--num_layers", type=int, default=30)
    ap.add_argument("--num_heads", type=int, default=12)
    ap.add_argument("--layers", default="0,15,29")
    ap.add_argument("--heads", default="0,5,11")
    ap.add_argument("--steps", default="0,25", help="要抓的采样步（0 = 第一步，最靠近纯噪声）")
    ap.add_argument("--branch", type=int, default=0, help="CFG 分支：0=条件分支，1=无条件分支")
    ap.add_argument("--branches", type=int, default=2, help="每个采样步的前向次数（CFG=2）")
    ap.add_argument("--max-queries", type=int, default=192, help="每层每头子采样多少行 query")
    ap.add_argument("--max-keys", type=int, default=0, help="0=保留全部 key（推荐）")
    ap.add_argument("--sparsities", default="0.5,0.7,0.8,0.9,0.95")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--target-err", type=float, default=0.02)
    ap.add_argument("--prompt", default="a static red cube on a table, plain background")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-sample-steps", type=int, default=0,
                    help="只跑前 N 步就停止（0=由 --steps 自动推断）")
    ap.add_argument("--outdir", default="captures")
    args = ap.parse_args(argv)

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    heads = [int(x) for x in args.heads.split(",") if x.strip()]
    steps = [int(x) for x in args.steps.split(",") if x.strip()]
    sparsities = [float(x) for x in args.sparsities.split(",") if x.strip()]
    max_steps = args.max_sample_steps or (max(steps) + 1)

    est_captures = len(layers) * len(heads) * len(steps)
    w, h = [int(v) for v in args.size.split("*")]
    latent_frames = (args.frame_num - 1) // 4 + 1
    n_tokens = latent_frames * ((w // 8) // 2) * ((h // 8) // 2)
    est_mb = est_captures * (n_tokens * 128 * 2 * 2) / (1024 ** 2)   # k,v 两个 fp16

    print("=== 抓取计划 ===")
    print("层: %s   头: %s   采样步: %s   CFG 分支: %d" % (layers, heads, steps, args.branch))
    print("序列长度估计: N ≈ %d token（%d latent 帧 × %d×%d patch 网格）"
          % (n_tokens, latent_frames, (w // 8) // 2, (h // 8) // 2))
    print("抓取组数: %d（层×头×步）  预计落盘: 约 %.0f MB（fp16）" % (est_captures, est_mb))
    print("只跑前 %d 个采样步，不生成完整视频" % max_steps)
    print("输出目录: %s" % os.path.abspath(args.outdir))

    if args.dry_run:
        print("\n[dry-run] 未加载模型。有卡时去掉 --dry-run 即可真正抓取。")
        print("建议顺序：先 --max-sample-steps 1 --layers 0 --heads 0 跑通，再扩大范围。")
        return 0

    os.makedirs(args.outdir, exist_ok=True)
    import torch

    mod, attr = find_attention_callable()
    print("\n找到注意力入口: %s.%s" % (mod.__name__, attr))
    original = getattr(mod, attr)

    state = {"call": 0}
    captures: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    def record(q, k, v, layer: int, step: int, branch: int) -> None:
        qb = to_bhld(q, args.num_heads)
        kb = to_bhld(k, args.num_heads)
        vb = to_bhld(v, args.num_heads)
        if qb is None or kb is None or vb is None:
            print("   [warn] 无法识别 q/k/v 布局，shape=%s/%s/%s" % (q.shape, k.shape, v.shape))
            return
        scale = 1.0 / math.sqrt(qb.shape[-1])
        g = torch.Generator().manual_seed(args.seed + layer * 131 + step)
        for head in heads:
            if head >= qb.shape[1]:
                continue
            n_keys = kb.shape[2]
            if args.max_keys and args.max_keys < n_keys:
                idx = torch.randperm(n_keys, generator=g)[: args.max_keys].sort().values
            else:
                idx = torch.arange(n_keys)
            rows = torch.randperm(qb.shape[2], generator=g)[: min(args.max_queries, qb.shape[2])]
            q_h = qb[0, head].index_select(0, rows.to(qb.device)).detach().float().cpu()
            k_h = kb[0, head].index_select(0, idx.to(kb.device)).detach().float().cpu()
            v_h = vb[0, head].index_select(0, idx.to(vb.device)).detach().float().cpu()
            s = (q_h @ k_h.transpose(-1, -2)) * scale
            delta = s - s.max(dim=-1, keepdim=True).values
            st = delta_stats(delta, sparsities, args.tau, args.target_err)
            st.update({"layer": layer, "head": head, "step": step, "branch": branch})
            summaries.append(st)
            captures.append({
                "layer": layer, "head": head, "step": step, "branch": branch,
                "q": q_h.half(), "k": k_h.half(), "v": v_h.half(), "delta": delta.half(),
                "q_rows": rows.tolist(), "k_idx": idx.tolist(), "scale": scale,
            })
            print("   捕获 layer=%d head=%d step=%d: N=%d, logit_std=%.3f, max_safe_sparsity=%.2f"
                  % (layer, head, step, delta.shape[1], st["logit_std"], st["max_safe_sparsity"]))

    def wrapper(q, k, v, *a, **kw):
        i = state["call"]
        layer = i % args.num_layers
        fwd = (i // args.num_layers) % args.branches
        step = i // (args.num_layers * args.branches)
        state["call"] = i + 1
        if layer in layers and fwd == args.branch and step in steps:
            with torch.no_grad():
                record(q, k, v, layer, step, fwd)
        return original(q, k, v, *a, **kw)

    setattr(mod, attr, wrapper)

    # 直接调 Wan 的推理入口，只跑少量步
    from wan import WanT2V

    class _A:
        pass

    a = _A()
    a.task = args.task
    a.ckpt_dir = args.ckpt_dir
    a.size = args.size
    a.frame_num = args.frame_num
    a.sample_steps = max_steps
    a.sample_shift = 8
    a.sample_guide_scale = 6
    a.offload_model = True
    a.t5_cpu = True
    a.use_prompt_extend = False
    a.device = "cuda" if torch.cuda.is_available() else "cpu"
    a.ulysses_size = 1
    a.ring_size = 1
    a.t5_fsdp = False
    a.dit_fsdp = False
    pipe = WanT2V(a)
    print("\n开始抓取（只跑 %d 步，不做 VAE 解码与写盘）..." % max_steps)
    try:
        pipe.generate(prompt=args.prompt, size=args.size, frame_num=args.frame_num,
                      seed=args.seed, t5_cpu=True, offload_model=True)
    except Exception as exc:  # noqa: BLE001
        print("   [info] 生成提前结束/报错（若已抓够数据可忽略）: %s" % exc)
    finally:
        setattr(mod, attr, original)

    # ---------------- 落盘 ----------------
    pt_path = os.path.join(args.outdir, "qkv_capture.pt")
    torch.save({"captures": captures,
                "meta": {"size": args.size, "frame_num": args.frame_num,
                         "layers": layers, "heads": heads, "steps": steps,
                         "seed": args.seed, "prompt": args.prompt}}, pt_path)

    csv_path = os.path.join(args.outdir, "capture_summary.csv")
    if summaries:
        keys: List[str] = []
        for s in summaries:
            for kk in s:
                if kk not in keys:
                    keys.append(kk)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            for s in summaries:
                wr.writerow(s)

    # ---------------- 人读报告 ----------------
    ranked = sorted(summaries, key=lambda r: -r["max_safe_sparsity"])
    lines = ["# 真实注意力抓取报告\n",
             "prompt: `%s`；size=%s；frame_num=%d；抓取步=%s\n" % (args.prompt, args.size, args.frame_num, steps)]
    if ranked:
        lines.append("\n## 可稀疏性排序（按 max_safe_sparsity 降序，误差目标 %.1f%%）\n"
                     % (args.target_err * 100))
        lines.append("| layer | head | step | N | logit_std | 熵(归一) | 顶1%质量 | max_safe_sparsity | rho_drop@0.9 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in ranked[:30]:
            lines.append("| %d | %d | %d | %d | %.3f | %.3f | %.3f | **%.2f** | %.3f |"
                         % (r["layer"], r["head"], r["step"], r["n_keys"], r["logit_std"],
                            r["attn_entropy_norm"], r["top1pct_mass"], r["max_safe_sparsity"],
                            r.get("rho_drop_s90", float("nan"))))
        med_std = float(sum(r["logit_std"] for r in ranked) / len(ranked))
        best = ranked[0]
        lines.append("\n## 结论与下一步\n")
        lines.append("- 抓取到的 logit std 中位数: **%.3f**（补偿有效区间约在 std ≲ 0.35）" % med_std)
        lines.append("- 最可稀疏的 (层,头): layer=%d head=%d step=%d，最大安全稀疏率约 %.2f"
                     % (best["layer"], best["head"], best["step"], best["max_safe_sparsity"]))
        if med_std <= 0.35:
            lines.append("- **判断：补偿在这个模型上有用武之地**。下一步做层内微基准（V1）与端到端配对评测。")
        else:
            lines.append("- **判断：多数头落在尖峰区间，补偿收益有限**。"
                         "建议把项目重心转向「按头选择稀疏策略」的边界分析，这也是有价值的结果。")
        if any(r.get("optimistic_flag") for r in ranked):
            lines.append("- 注意：部分配置的 rho_drop > 20%，此时误差代理是乐观估计（实测低估 1.5~3 倍）。")
    lines.append("\n## 下游用法\n")
    lines.append("```bash")
    lines.append("python feasibility.py --delta-file <导出的 delta.npy> --qkv-pt qkv_capture.pt")
    lines.append("```")
    md_path = os.path.join(args.outdir, "capture_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n写出: %s\n      %s\n      %s" % (pt_path, csv_path if summaries else "(无 summary)", md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
