"""视频生成质量 / 闪烁评测工具集（配对比较专用）。

模块划分：
  videoutil         : 视频解码、探测、完整性校验（防 cache 复用与静默失败）
  flicker           : 自研闪烁指标（背景区域帧间 L1 / 光流残差 / 分块 DC 方差）
  proxy_consistency : 轻量代理一致性指标（CLIP / DINO 特征），用于快速迭代
  vbench_adapter    : 封装官方 VBench 的 background/subject consistency、motion smoothness
  fvd               : FVD / KID（小规模补充指标 + bootstrap 置信区间）
  paired            : 配对统计（bootstrap CI、Wilcoxon、效应量、达标判定）
  report            : 生成 markdown 报告与图（效率-质量前沿曲线等）
"""

__all__ = [
    "videoutil",
    "flicker",
    "proxy_consistency",
    "vbench_adapter",
    "fvd",
    "paired",
    "report",
]

__version__ = "0.1.0"
