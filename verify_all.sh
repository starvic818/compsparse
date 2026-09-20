#!/usr/bin/env bash
# 一键自检：不需要 GPU，验证仓库里的工具链在当前环境下能不能用。
#
# 用法：
#   bash verify_all.sh            # 全部检查
#   bash verify_all.sh --quick    # 跳过较慢的相图/标定
#
# 期望结果：最后一行输出 "ALL PASS"。
# 任何一项 FAIL 都不影响其他项继续跑，便于一次性看到所有问题。

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

PASS=0
FAIL=0
FAILED_NAMES=""

run() {
  local name="$1"
  shift
  printf '\n=== %s ===\n' "$name"
  if "$@"; then
    printf '[PASS] %s\n' "$name"
    PASS=$((PASS + 1))
  else
    printf '[FAIL] %s\n' "$name"
    FAIL=$((FAIL + 1))
    FAILED_NAMES="${FAILED_NAMES}${name}; "
  fi
}

OUT="results_verify"
mkdir -p "$OUT"

printf '仓库根目录: %s\n' "$ROOT"
printf 'Python: %s\n' "$(python -V 2>&1)"
printf 'OpenCV: %s\n' "$(python -c 'import cv2; print(cv2.__version__)' 2>&1 | tail -1)"
printf 'PyTorch: %s\n' "$(python -c 'import torch; print(torch.__version__, "cuda=" + str(torch.cuda.is_available()))' 2>&1 | tail -1)"

run "1. 稀疏注意力代数与误差界（11 项测试）" \
    python sparse_attn_proto/test_sparse_attn.py

run "2. 评测套件自检（合成闪烁视频组）" \
    python video_quality_eval/selftest.py

run "3. 生成实验配置（16 prompt / 96 runs）" \
    python run_scripts/make_experiment.py --outdir "$OUT/manifests" --video-dir ./works

run "4. sweep dry-run（不加载模型）" \
    python run_scripts/sweep.py --config "$OUT/manifests/sweep_config.json" \
        --outdir "$OUT/sweep" --dry-run --limit 3

run "5. capture dry-run（抓取计划）" \
    python run_scripts/capture_qkv.py --dry-run --max-sample-steps 1 --layers 0 --heads 0 --steps 0

run "6. 环境基线采集" \
    python video_quality_eval/env_report.py --outdir "$OUT/env"

if [ "$QUICK" -eq 0 ]; then
  run "7. 可行性相图（合成 logits）" \
      python sparse_attn_proto/feasibility.py --synthetic --outdir "$OUT/feasibility"

  run "8. 误差代理标定" \
      python sparse_attn_proto/feasibility.py --calibrate --n 192 --d 32 \
          --sparsities 0.5,0.8,0.9 --outdir "$OUT/calibration"
fi

printf '\n============================\n'
printf 'PASS: %d   FAIL: %d\n' "$PASS" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  printf '失败项: %s\n' "$FAILED_NAMES"
  exit 1
fi
printf 'ALL PASS\n'
printf '产物目录: %s（已被 .gitignore 忽略）\n' "$OUT"
