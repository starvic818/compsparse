"""CPU 单元测试：验证稀疏注意力代数实现与理论界。

这些测试覆盖了"上卡之前必须确认正确"的全部内容：
  1. 稠密参考实现正确（对得上 softmax）
  2. 一阶矩 M_A / S_A / G_A 与逐元素暴力求和一致（代数没写错）
  3. 误差上界成立（理论命题 1/2 在真实数据上不被违反）
  4. 窄带内一阶补偿优于硬丢弃；宽带内一阶补偿反而更差
     —— 这两条一起构成"必须分段"的实证依据
  5. 分母二阶确实优于分母一阶
  6. break-even 点就在 |delta| = 1（此时一阶权重 1+delta 恰好为 0）

运行：python test_sparse_attn.py     （纯 CPU，几秒）
"""

from __future__ import annotations

import math
import sys

import torch

from sparse_attn import (
    attention_bundle,
    break_even_delta,
    per_key_relative_error,
    rel_error,
    summarize_errors,
)

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print("%-58s %s%s" % (name, "PASS" if ok else "FAIL", ("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(name)


def toy(n: int = 192, d: int = 32, seed: int = 0, k_scale: float = 1.0):
    """k_scale 控制 logits 的散布：
       k_scale ~ 1.0  -> logits 分散（近似"真实模型里某些头"的尖峰分布，A 段往往为空）
       k_scale ~ 0.3  -> logits 压缩（近似"某些头注意力很平"的分布，A 段被填满）
    两种区间对应完全不同的结论，见 test_spread_logits_* 与 test_compressed_logits_*。
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, d, generator=g)
    k = torch.randn(n, d, generator=g) * k_scale
    v = torch.randn(n, d, generator=g)
    return q, k, v


def test_dense_reference():
    q, k, v = toy()
    b = attention_bundle(q, k, v, exact_top_m=8, delta_hi=0.4)
    s = (q @ k.T) / math.sqrt(q.shape[-1])
    ref = torch.softmax(s, dim=-1) @ v
    err = float(rel_error(b["out"]["dense"], ref).max())
    check("1) 稠密参考实现 == softmax(s)@v", err < 1e-6, "max_rel_err=%.2e" % err)


def test_moments_bruteforce():
    q, k, v = toy(n=64, d=16, seed=1)
    b = attention_bundle(q, k, v, exact_top_m=6, delta_hi=0.35)
    A = b["masks"]["A"]
    M_A = b["moments"]["M_A"]
    S_A = b["moments"]["S_A"]
    V_A = b["moments"]["V_A"]
    G_A = b["moments"]["G_A"]

    worst = 0.0
    for i in (0, 7, 33):
        idx = torch.nonzero(A[i], as_tuple=False).flatten()
        M_ref = torch.zeros_like(M_A[i])
        S_ref = torch.zeros_like(S_A[i])
        V_ref = torch.zeros_like(V_A[i])
        G_ref = torch.zeros_like(G_A[i])
        for j in idx.tolist():
            M_ref += torch.outer(k[j], v[j])
            G_ref += torch.outer(k[j], k[j])
            S_ref += k[j]
            V_ref += v[j]
        worst = max(
            worst,
            float((M_A[i] - M_ref).abs().max()),
            float((G_A[i] - G_ref).abs().max()),
            float((S_A[i] - S_ref).abs().max()),
            float((V_A[i] - V_ref).abs().max()),
        )
    check("2) 一阶/二阶矩与暴力求和一致", worst < 1e-5, "max_abs_diff=%.2e" % worst)


def test_bound_holds():
    """实测 A 段相对误差必须不超过理论界 (Σ r_j ||v_j||)/||n_A||。"""
    q, k, v = toy(seed=2)
    ok = True
    detail = []
    for delta_hi in (0.1, 0.3, 0.6, 1.2):
        b = attention_bundle(q, k, v, exact_top_m=8, delta_hi=delta_hi)
        n_true = b["true"]["n_A"]
        n_apx = b["approx"]["n_A_1"]
        denom = n_true.norm(dim=-1).clamp_min(1e-12)
        meas = ((n_apx - n_true).norm(dim=-1) / denom).median()
        bound = (b["theory"]["bound_numerator"] / denom).median()
        detail.append("d=%.1f:%.3f/%.3f" % (delta_hi, float(meas), float(bound)))
        if float(meas) > float(bound) * 1.0001:
            ok = False
    check("3) 实测误差 <= 理论上界（全部 delta_hi）", ok, " ".join(detail))


def test_narrow_band_order1_beats_drop():
    q, k, v = toy(seed=3, k_scale=0.3)      # 压缩 logits：A 段非空
    b = attention_bundle(q, k, v, exact_top_m=8, delta_hi=0.3)
    e1 = float(rel_error(b["out"]["order1"], b["out"]["dense"]).median())
    e0 = float(rel_error(b["out"]["hard"], b["out"]["dense"]).median())
    n_band = float(b["masks"]["A"].sum(dim=-1).float().median())
    check("4a) 窄带(|d|<=0.3)：一阶补偿优于硬丢弃", e1 < e0,
          "order1=%.4f < hard=%.4f (A段中位 %d 个 key)" % (e1, e0, n_band))


def test_wide_band_order1_worse_than_drop():
    """一阶泰勒在长尾上会给出负权重，因此比直接丢弃更差 —— 这就是必须有 D 段的原因。"""
    q, k, v = toy(seed=3)
    b = attention_bundle(q, k, v, exact_top_m=8, delta_hi=8.0)
    e1 = float(rel_error(b["out"]["order1"], b["out"]["dense"]).median())
    e0 = float(rel_error(b["out"]["hard"], b["out"]["dense"]).median())
    check("4b) 宽带(delta_hi=8)：一阶补偿反而差于硬丢弃", e1 > e0,
          "order1=%.4f > hard=%.4f" % (e1, e0))


def test_denominator_second_order_helps():
    q, k, v = toy(seed=4, k_scale=0.3)
    b = attention_bundle(q, k, v, exact_top_m=8, delta_hi=0.5)
    z_true = b["true"]["z_A"]
    e1 = ((b["approx"]["z_A_1"] - z_true).abs() / z_true.clamp_min(1e-12)).median()
    e2 = ((b["approx"]["z_A_2"] - z_true).abs() / z_true.clamp_min(1e-12)).median()
    check("5) 分母二阶优于分母一阶", float(e2) < float(e1),
          "z_err1=%.4f > z_err2=%.4f" % (float(e1), float(e2)))


def test_exact_budget_reduces_error():
    q, k, v = toy(seed=5)
    prev = None
    mono = True
    for m in (4, 16, 64):
        b = attention_bundle(q, k, v, exact_top_m=m, delta_hi=0.4)
        e = float(rel_error(b["out"]["order1"], b["out"]["dense"]).median())
        if prev is not None and e > prev * 1.05:
            mono = False
        prev = e
    check("6) 精确预算越大误差越小（单调）", mono)


def test_perfect_when_everything_exact():
    q, k, v = toy(n=48, seed=6)
    b = attention_bundle(q, k, v, exact_top_m=48, delta_hi=0.4)
    errs = [
        float(rel_error(b["out"][n], b["out"]["dense"]).max())
        for n in ("hard", "order1", "hybrid")
    ]
    check("7) 全精确时三种策略均等于 dense", max(errs) < 1e-5, "max=%.2e" % max(errs))


def test_break_even_point():
    be = break_even_delta()
    check("8) 一阶/丢弃 break-even 在 |delta|=1", abs(be - 1.0) < 2e-3, "solved=%.4f" % be)
    rel = float(per_key_relative_error(torch.tensor([-0.5])))
    # 精确值 1 - e^0.5 * 0.5 = 0.1756；理论上界 (d^2/2)e^d = 0.2062（界成立且不紧）
    check("9) |delta|=0.5 单 key 相对误差 = 0.1756 且 < 上界 0.2062",
          abs(rel - 0.1756) < 2e-3 and rel < 0.2062, "=%.4f" % rel)


def test_spread_logits_band_empty():
    """分散 logits（k_scale≈1）：A 段几乎为空 -> 补偿退化为硬丢弃，且误差巨大。
    含义：稀疏率不是自由参数，它由注意力分布决定。"""
    q, k, v = toy(seed=8, k_scale=1.0)
    b = attention_bundle(q, k, v, exact_top_m=16, delta_hi=0.4)
    n_band = float(b["masks"]["A"].sum(dim=-1).float().median())
    rho_drop = float(b["flow"]["rho_drop"].median())
    e0 = float(rel_error(b["out"]["hard"], b["out"]["dense"]).median())
    check("10) 分散 logits：A 段为空且硬丢弃误差>50%%",
          n_band < 5 and rho_drop > 0.5 and e0 > 0.5,
          "A段=%d rho_drop=%.2f rel_err=%.2f" % (n_band, rho_drop, e0))


def test_compressed_logits_band_helps():
    """压缩 logits（k_scale≈0.05，近似"平坦注意力"）：A 段被填满 -> 补偿把误差压掉三个数量级。"""
    q, k, v = toy(seed=8, k_scale=0.05)
    b = attention_bundle(q, k, v, exact_top_m=16, delta_hi=0.4)
    n_band = float(b["masks"]["A"].sum(dim=-1).float().median())
    e1 = float(rel_error(b["out"]["order1"], b["out"]["dense"]).median())
    e0 = float(rel_error(b["out"]["hard"], b["out"]["dense"]).median())
    check("11) 压缩 logits：A 段非空且一阶补偿显著降低误差",
          n_band > 100 and e1 < 0.01 * e0,
          "A段=%d order1=%.5f vs hard=%.4f (gain=%.0fx)" % (n_band, e1, e0, e0 / max(e1, 1e-12)))


def main() -> int:
    torch.manual_seed(0)
    test_dense_reference()
    test_moments_bruteforce()
    test_bound_holds()
    test_narrow_band_order1_beats_drop()
    test_wide_band_order1_worse_than_drop()
    test_denominator_second_order_helps()
    test_exact_budget_reduces_error()
    test_perfect_when_everything_exact()
    test_break_even_point()
    test_spread_logits_band_empty()
    test_compressed_logits_band_helps()

    q, k, v = toy(n=256, seed=7)
    print("\n--- 参考数值（N=256, d=32, 随机 logits, delta_hi=0.4, top_m=16）---")
    b = attention_bundle(q, k, v, exact_top_m=16, delta_hi=0.4)
    for kk, vv in sorted(summarize_errors(b).items()):
        print("  %-32s %.5f" % (kk, vv))

    print("\n结论:", "ALL PASS" if not FAILURES else ("FAIL: " + ", ".join(FAILURES)))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
