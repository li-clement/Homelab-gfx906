# Homelab GFX906 - MI50 AI 实验环境

> **[ 🇬🇧 English Version ](./readme.md)**

本目录是基于 `gfx906-ml` 项目的扩展，专门针对 **AMD Radeon Instinct MI50 (gfx906)** 计算卡优化的 AI 实验环境。

## 🖥️ 硬件配置 (Hardware Spec)

本仓库的脚本与参数调优基于以下工作站环境：

* **主机**: [Lenovo ThinkStation P620](https://psref.lenovo.com/syspool/Sys/PDF/ThinkStation/ThinkStation_P620/ThinkStation_P620_Spec.pdf)
* **CPU**: AMD Ryzen™ Threadripper™ PRO 5945WX (12核/24线程)
* **内存**: 128GB DDR4 RDIMM ECC
* **显卡**: AMD Radeon Instinct MI50 32GB (HBM2)
* **存储**:
    * **系统盘**: 512GB NVMe SSD
    * **数据/缓存**: 1TB SATA SSD
    * **冷存储**: 6TB SAS HDD

## 🚀 核心功能

该环境集成了以下核心功能：
1.  **LLM 推理与数据合成**：使用 vLLM 和 Distilabel 进行高性能推理及数据重写。
2.  **多模态绘图**：运行 GLM-Image 等扩散模型进行文生图。
3.  **模型微调**：使用 LlamaFactory 对 Qwen 等大模型进行 LoRA 微调。
4.  **集群部署**：提供基于 K8s/K3s 的 Jupyter 与 vLLM 环境部署方案。

## 📂 目录结构

```text
.
├── homelab/            # 本地实验 Notebooks
│   ├── DataGen.ipynb   # [推理] vLLM 服务部署与 Distilabel 数据生成/重写
│   ├── Omni.ipynb      # [绘图] GLM-Image 环境构建与图像生成 (解决依赖冲突)
│   └── finetune.ipynb  # [训练] LlamaFactory 模型微调 (针对 MI50 32G 显存优化)
└── k8s/                # Kubernetes/K3s 集群部署
    ├── Dockerfile      # 集成 Jupyter, vLLM 及 ROCm 环境的镜像构建文件
    └── vllm-finetune-deploy.yaml # K8s 部署清单 (Manifest)

```

## 🛠️ 使用指南

### 1. 本地实验 (`homelab/`)

* **LLM 推理 (`DataGen.ipynb`)**：在 MI50 上本地部署 OpenAI 兼容 API，并使用 `Distilabel` 构建自动化数据处理流水线。包含用于“反抄袭/重构”的专用提示词系统。
* **多模态绘图 (`Omni.ipynb`)**：通过创建独立虚拟环境 (`env_glm`) 并锁定 Numpy 版本，解决了 ROCm 环境下的依赖冲突，成功运行 **GLM-Image**。内置防止 MI50 黑图（VAE float32）的补丁。
* **模型微调 (`finetune.ipynb`)**：针对 MI50 (32GB) 深度定制的 **LlamaFactory** 训练流程。强制开启 `fp16`（硬件不支持 bf16），并优化了 Batch Size 和梯度累积步数以确保稳定性。

### 2. Kubernetes / K3s 集群部署 (`k8s/`)

支持将完整的 AI 开发环境（Jupyter Lab + vLLM）部署到 K8s 或 K3s 集群中。

* **构建镜像**：
`k8s/Dockerfile` 包含了完整的运行环境，预装了 ROCm 依赖、vLLM 推理引擎及 Jupyter Lab。
```bash
cd k8s
docker build -t your-registry/gfx906-lab:latest .

```


* **部署服务**：
使用提供的 YAML 清单部署 Pod。该文件已配置好 GPU 资源请求及必要的存储挂载。
```bash
kubectl apply -f k8s/vllm-finetune-deploy.yaml

```


*部署完成后，你将获得一个集成了开发（Jupyter）与推理服务（vLLM）的 Pod。*

## ⚠️ 常见问题与注意事项

1. **MI50 不支持 BF16**：MI50 (Vega 20) 无法进行硬件级 `bfloat16` 加速。在微调脚本中，配置已强制设为 `"fp16": True`。
2. **Numpy 版本冲突**：`Omni.ipynb` 通过独立 venv 锁定 `numpy` 来解决 Diffusers 兼容性问题。
3. **显存管理**：在切换不同任务（如从 vLLM 切换到微调）时，建议运行 Notebook 中的清理命令或重启内核，以防止显存碎片化。