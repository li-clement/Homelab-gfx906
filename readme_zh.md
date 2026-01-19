# Homelab GFX906 - MI50 AI 实验环境

> **[ 🇬🇧 English Version ](./readme.md)**

本目录 (`homelab/`) 是基于 `gfx906-ml` 项目的扩展，专门针对 **AMD Radeon Instinct MI50 (gfx906)** 计算卡优化的 AI 实验环境。

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

## 📂 目录结构

```text
homelab/
├── DataGen.ipynb   # [推理] vLLM 服务部署与 Distilabel 数据生成/重写
├── Omni.ipynb      # [绘图] GLM-Image 环境构建与图像生成 (解决依赖冲突)
└── finetune.ipynb  # [训练] LlamaFactory 模型微调 (针对 MI50 32G 显存优化)

```

## 🛠️ 脚本用法详解

### 1. LLM 推理与数据生成 (`DataGen.ipynb`)

演示了如何在 MI50 上本地部署 OpenAI 兼容的 API 服务，并构建自动化流水线来处理/生成文本数据。

* **核心组件**：`vLLM`, `Distilabel`, `Qwen2.5/Qwen3`
* **主要流程**：
1. **环境检查**：自动检测 `vLLM` 和 `PyTorch` (ROCm 版本) 安装状态。
2. **启动服务**：通过子进程启动 `vLLM` 本地服务器，加载模型（显存利用率 0.95，上下文 8192）。
3. **构建流水线**：使用 `Distilabel` 连接本地 API。
4. **深度重写**：定义特殊的“反抄袭/重构”提示词，对输入文本进行深度改写。
5. **结果输出**：生成 `deep_rewritten_results.json`。



> **💡 提示**：运行服务启动代码块后，请确保看到 "Uvicorn running" 字样后再运行后续单元格。

### 2. 多模态绘图 (`Omni.ipynb`)

解决了 ROCm 环境下复杂的依赖冲突（特别是 Numpy 版本问题），成功运行 **GLM-Image** 模型。

* **核心组件**：`Diffusers`, `GLM-Image`, `Virtualenv`
* **关键特性 (环境隔离)**：创建 `env_glm` 虚拟环境，强制安装 `numpy==1.26.4` 以兼容绘图库，同时通过 `--system-site-packages` 继承系统级 ROCm PyTorch。
* **MI50 专项优化**：
* **防黑图补丁**：自动将 VAE/VQModel 转换为 `float32`（MI50 半精度解码 VAE 会导致黑图）。
* **CPU Offload**：开启模型 CPU 卸载以适应 32GB 显存。


* **使用方法**：运行环境构建代码 -> **重启内核 (Restart Kernel)** -> 切换为 `env_glm` 内核 -> 运行生成。

### 3. 模型微调 (`finetune.ipynb`)

基于 LlamaFactory 进行高效 LoRA 微调，配置文件针对 MI50 硬件特性进行了深度定制。

* **核心组件**：`LlamaFactory`, `Deepspeed`
* **硬件优化配置 (32GB VRAM)**：
* **Flash Attention**：启用 SDPA 加速。
* **精度要求**：**必须开启 `fp16**`。MI50 (Vega 20) 硬件不支持 bf16 加速，严禁使用。
* **吞吐优化**：`batch_size=4`，`grad_accum=4` (等效 Batch 16)，`cutoff_len=4096`。


* **流程**：显存清理 -> 数据注册 (`evol_instruct_dataset.json`) -> 配置生成 -> 启动训练。

## ⚠️ 常见问题与注意事项

1. **MI50 不支持 BF16**：MI50 无法进行硬件级 `bfloat16` 加速。在 `finetune.ipynb` 中，配置已强制设为 `"fp16": True`。强行使用 bf16 会导致报错或回退到极慢的软件模拟。
2. **Numpy 版本冲突**：`Omni.ipynb` 通过独立 venv 锁定 `numpy==1.26.4` 来解决 Diffusers 兼容性，请勿在系统全局环境中随意升级 Numpy。
3. **显存管理**：在切换不同任务（如从 vLLM 切换到微调）时，建议运行 Notebook 中的清理命令或重启内核，以防止显存碎片化导致 OOM。