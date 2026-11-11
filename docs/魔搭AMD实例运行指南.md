# 魔搭（ModelScope）AMD / ROCm 实例运行指南

适用镜像示例：`ubuntu22.04-rocm7.2.3-py312-torch2.12.0-...`  
该环境是 **AMD GPU + ROCm**，不是 NVIDIA CUDA。PyTorch ROCm 版仍通过 `torch.cuda` / 设备名 `cuda` 访问显卡。

---

## 0. 启动实例

在魔搭控制台点击「启动」，等状态变为运行中，再打开终端。

---

## 1. 拉取仓库

```bash
git clone https://github.com/Ljt-lj/predictive-coding-local-learning.git
cd predictive-coding-local-learning
```

若仓库尚无已推送最新代码，也可上传/同步本地目录到实例后再 `cd` 进去。

---

## 2. 安装依赖（关键：不要重装 torch）

镜像已预装 ROCm 版 PyTorch。只装实验额外依赖：

```bash
python -c "import torch; print(torch.__version__, 'cuda?', torch.cuda.is_available(), 'hip=', getattr(torch.version,'hip',None))"
pip install -r requirements-platform.txt
```

**禁止**执行 `pip install torch` / `pip install -r requirements.txt`（会把 ROCm 轮子盖成 CPU/CUDA 轮子，显卡就废了）。

---

## 3. 确认显卡可见

```bash
python scripts/check_device.py
```

期望看到类似：

- `cuda_available=True`
- `backend=rocm/hip (...)`
- `matmul_ok=True`
- 最后一行 `OK`

若失败：检查是否误装了 CPU 版 torch；可重开实例或重装镜像环境。

---

## 4. 冒烟测试（建议先跑）

```bash
python scripts/smoke_test.py --device cuda
# 或：python scripts/smoke_test.py --device rocm
```

`--device rocm` / `gpu` / `hip` 在本项目里都会映射到 `cuda` 设备。

---

## 5. 完整实验（推荐）

一键：BP + PCN(T=20/50/100) + 出图：

```bash
python scripts/run_all.py --cloud --device cuda
```

`--cloud` 使用更大 `batch_size=512`、`num_workers=4` 的配置，更适合这台大显存实例。

可选缩短 epoch：

```bash
python scripts/run_all.py --cloud --device cuda --epochs 5
```

分步跑也可以：

```bash
python scripts/run_bp.py  --config configs/cloud_bp.yaml --device cuda
python scripts/run_pcn.py --config configs/cloud_pcn_t20.yaml --device cuda
python scripts/run_pcn.py --config configs/cloud_pcn_t50.yaml --device cuda
python scripts/run_pcn.py --config configs/cloud_pcn_t100.yaml --device cuda
python scripts/make_figures.py --runs-dir runs --out-dir figures
```

---

## 6. 结果在哪

| 路径 | 内容 |
|------|------|
| `runs/<run_id>/` | `metrics.json`、`history.csv`、`energy.csv`、`model.pt` |
| `figures/` | PPT 用四张图 |

下载回本地后放进 PPT 即可。

---

## 常见问题

1. **`torch.cuda.is_available() == False`**  
   多半重装了错误的 torch。不要用 `requirements.txt` 装 torch。

2. **PCN 仍然偏慢**  
   正常：每个 batch 有推理迭代。可先 `--epochs 5`，或先只跑 `cloud_pcn_t50`。

3. **设备名为什么是 cuda？**  
   ROCm 版 PyTorch 的兼容层如此设计，不是装错成 NVIDIA 驱动。
