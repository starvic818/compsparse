# 建仓库时的选择建议

## 1. Description（GitHub 仓库描述框，≤350 字符）

推荐（英文，便于检索）：

```text
Piecewise sparse attention for long-video diffusion transformers: exact compute where it matters,
Taylor compensation where the tail is negligible, plus a measurement-first toolkit for sparsity
budgets and background-flicker evaluation. (Wan2.1 / DiT)
```

更短一版：

```text
Compensated sparse attention for long-video diffusion transformers — exact + Taylor-compensated +
dropped, with distribution-aware sparsity analysis and paired quality/flicker evaluation.
```

中文版（如果你想让人一眼看懂）：

```text
补偿式稀疏注意力：面向长视频扩散模型的「精确计算 + 泰勒补偿 + 尾部丢弃」三段式方案，
附带可稀疏性分析、背景闪烁度量与批量实验工具链。
```

## 2. Topics（About 右侧的标签）

```text
sparse-attention  video-diffusion  diffusion-transformer  flash-attention
long-video  inference-acceleration  wan2.1  pytorch  vbench
```

## 3. 可见度（Visibility）

```text
开发期：Private（推荐）—— 代码还在动，且报告未定稿
答辩/提交前：一键转 Public —— 你的预期目标里写了"形成一套可运行的代码库"，
             公开可验证比截图更有说服力
红线：Compshare 实例密码、SSH 私钥、任何 API token 绝不能进仓库（.gitignore 已覆盖）
```

## 4. 添加 README / .gitignore / 许可 —— 关键提醒

⚠️ **如果你本地已经有一个文件夹要推上去，不要在 GitHub 上勾 README / .gitignore / License。**
那会在远端产生一个初始提交，而本地是另一条历史，直接 `git push` 会被拒绝（non-fast-forward）。

两条正确路径，选一条：

### 路径 A：本地已有文件夹（推荐）

```text
1) GitHub 建空仓库：README / .gitignore / License 全部不勾
2) 本地：
   cd <你的项目目录>
   git init
   git add .
   git commit -m "compsparse: offline deliverables (derivation, prototype, eval toolkit)"
   git branch -M main
   git remote add origin https://github.com/<你>/compsparse.git
   git push -u origin main
3) 之后把本目录的 README.md / .gitignore / LICENSE 放进项目根目录，再提交一次
```

### 路径 B：你已经勾了 README / .gitignore / License

```text
1) 不要直接推本地旧仓库，改成克隆远端：
   git clone https://github.com/<你>/compsparse.git
   cd compsparse
2) 把你本地的文件复制进来（覆盖或合并 README.md、.gitignore）
3) git add . && git commit -m "add offline deliverables" && git push
```

## 5. 许可（License）怎么选

| 选择 | 适合 | 说明 |
|---|---|---|
| **Apache-2.0** | 本项目的推荐 | 与基座模型 Wan2.1（Apache-2.0）一致；含专利授权，企业/学术都友好；要求保留 NOTICE 与变更说明 |
| MIT | 想最省事 | 最宽松、最短，但无专利条款 |
| GPL-3.0 | ❌ 不建议 | 传染性强，与你要集成的生态（Wan2.1、VBench 依赖）容易冲突 |
| Unlicense / CC0 | ❌ 不建议 | 对专利与免责声明不明确，学术项目不宜 |

**建议直接用 GitHub 的 License 模板生成**（选择 Apache License 2.0，它会自动填入年份与作者），
不要手抄许可证文本 —— 文本不完整等于没有许可证。

## 6. 提交前自查

```bash
git status                    # 确认没有 results/ captures/ *.pt *.mp4 被加入
du -sh .git                   # 仓库体积应远小于 1MB（代码 + 文档）
git log --oneline | head      # 提交信息能看懂
grep -rn "password\|token" .  # 确认没有凭据（配合 .gitignore）
```
