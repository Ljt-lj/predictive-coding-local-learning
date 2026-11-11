# Predictive Coding vs Backprop on MNIST

课程《类脑智能与人工通用智能》期末实验代码：对比 **Predictive Coding Network (PCN)** 与 **Backpropagation (BP)** 在 MNIST 上的分类表现。

配套文档：

- [`实验方案.md`](./实验方案.md) — 课程实验方案
- [`技术设计文档.md`](./技术设计文档.md) — 工程落地细则

## 环境

- Python ≥ 3.10

**本机（Windows / 无独显）：**

```bash
# 先按 pytorch.org 安装匹配本机的 torch / torchvision，再：
pip install -r requirements.txt
```

**魔搭 AMD / ROCm 免费实例（推荐跑完整实验）：**

镜像已预装 ROCm 版 PyTorch。**不要**再 `pip install torch`，否则会覆盖 ROCm 轮子。

```bash
pip install -r requirements-platform.txt
python scripts/check_device.py
python scripts/run_all.py --cloud --device cuda
```

详细步骤见 [`docs/魔搭AMD实例运行指南.md`](./docs/魔搭AMD实例运行指南.md)。  
说明：ROCm 下设备名仍是 `cuda`（PyTorch HIP 兼容层）。

## 快速冒烟测试

```bash
python scripts/smoke_test.py
# GPU / ROCm：
python scripts/smoke_test.py --device cuda
```

默认使用 MNIST 子集（train 2000 / test 500）、2 个 epoch，先验证 BP 与 PCN 全流程可跑通。

## 完整实验

```bash
# 对照组 BP
python scripts/run_bp.py --config configs/bp.yaml

# PCN 三组 T_infer
python scripts/run_pcn.py --config configs/pcn_t20.yaml
python scripts/run_pcn.py --config configs/pcn_t50.yaml
python scripts/run_pcn.py --config configs/pcn_t100.yaml

# 或一键跑完并出图
python scripts/run_all.py

# 魔搭大显存实例推荐：
python scripts/run_all.py --cloud --device cuda
```

### 日志与断点续跑

每个 run 目录会写入：

- `train.log` — 训练日志（控制台同步）
- `checkpoint.pt` — 每个 epoch 后的断点（完成后删除）
- `status.json` — `running` / `completed`
- `history.csv` / `metrics.json` / `model_best.pt`

```bash
# 续跑最近一次未完成的同配置实验
python scripts/run_pcn.py --config configs/pcn_t50.yaml --resume
python scripts/run_bp.py --config configs/bp.yaml --resume

# 指定某个 run 目录续跑
python scripts/run_pcn.py --run-dir runs/pcn_t50_20261111_162712

# 全量套件：已完成的跳过，未完成的从断点继续
python scripts/run_all.py --resume
python scripts/run_all.py --cloud --device cuda --resume
```

出图：

```bash
python scripts/make_figures.py --runs-dir runs --out-dir figures
```

生成：

- `figures/fig1_acc_curves.png`
- `figures/fig2_t_infer_bar.png`
- `figures/fig3_energy_curve.png`
- `figures/fig4_time_bar.png`

## 架构

- 层宽对齐：`784 → 256 → 128 → 10`
- BP：ReLU，输入 `[0,1]`
- PCN：`tanh`，输入映射到 `[-1,1]`（`pcn-torch` 生成式 PC 在 ReLU 下易卡在随机水平；层宽仍与 BP 对齐）
- PCN 使用 [`pcn-torch`](https://pypi.org/project/pcn-torch/)（局部推理 + 局部学习；测试时不 clamp 标签）

## 结果目录

每次运行写入 `runs/<run_id>/`：

- `config.yaml`
- `history.csv`
- `metrics.json`
- `energy.csv`（仅 PCN）
- `model.pt`
