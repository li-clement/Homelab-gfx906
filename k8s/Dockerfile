# =========================================================
# 基础镜像：AMD ROCm vLLM (官方适配 gfx906)
# =========================================================
FROM docker.io/mixa3607/vllm-gfx906:0.11.0-rocm-6.3.3

ENV DEBIAN_FRONTEND=noninteractive

# 1. 系统依赖
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    curl git vim wget unzip \
    ffmpeg libsm6 libxext6 \
    p7zip-full imagemagick \
    openjdk-11-jdk \
    build-essential \
    libsamplerate0-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Python 配置
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 核心基础库 (版本控制中心)
# ---------------------------------------------------------------------------
# 策略：优先满足 vLLM (底层)，强迫上层应用适配
# vLLM 要求: pydantic >= 2.11.7
# LLaMA-Factory 要求: tyro < 0.9.0, numpy < 2.0.0
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    "pydantic>=2.11.7" \
    "tyro<0.9.0" \
    "ray[data,default]" \
    loguru \
    pyyaml \
    hf-transfer

# 4. Data-Juicer 安装
RUN pip install --no-cache-dir py-data-juicer --no-deps

# 5. Data-Juicer 依赖补齐
RUN pip install --no-cache-dir \
    # 再次强调版本约束，防止被 data-juicer 依赖带偏
    "numpy<2.0.0" \
    "pydantic>=2.11.7" \
    "tyro<0.9.0" \
    "av==13.1.0" \
    "emoji==2.2.0" \
    "fsspec==2023.5.0" \
    "spacy==3.8.7" \
    "samplerate==0.1.0" \
    # 补齐依赖
    bs4 gitpython lz4 "mcp[cli]>=1.7.0" mwparserfromhell \
    plotly pylance python-docx resampy seaborn streamlit tomli-w uv \
    jupyterlab notebook ipywidgets matplotlib \
    pandas scipy \
    dill jsonlines ujson multiprocess tqdm \
    transformers "datasets<3.0.0" \
    simple-di click requests tomli regex packaging platformdirs typing-extensions pyarrow \
    pdfplumber zstandard wget librosa jsonargparse tabulate wordcloud

# 6. 微调组件
WORKDIR /app
RUN pip install --no-cache-dir triton deepspeed
RUN pip install --no-cache-dir https://codeload.github.com/ROCm/bitsandbytes/tar.gz/refs/heads/rocm_enabled
# LLaMA-Factory
RUN git clone --depth 1 https://githubfast.com/hiyouga/LLaMA-Factory.git
WORKDIR /app/LLaMA-Factory

# ⚠️⚠️ 核心黑魔法：修改 LLaMA-Factory 源码依赖 ⚠️⚠️
# 1. 删除 torch (防覆盖 ROCm)
# 2. 删除 pydantic 限制 (解决与 vLLM 的冲突)
# 3. 删除 tyro 限制 (我们要手动控制 tyro)
RUN sed -i '/torch/d' requirements.txt && \
    sed -i '/torchvision/d' requirements.txt && \
    sed -i '/pydantic/d' requirements.txt && \
    sed -i '/tyro/d' requirements.txt

# 安装 LLaMA-Factory (此时 requirements 已经干净了，不会报错)
RUN pip install --no-deps -e .

# LLaMA-Factory 依赖补齐
# 注意：这里我们安装高版本 pydantic，同时保持其他包兼容
RUN pip install --no-cache-dir \
    "numpy<2.0.0" \
    "pydantic>=2.11.7" \
    "tyro<0.9.0" \
    "gradio>=4.38.0,<=5.45.0" \
    "trl>=0.8.6,<=0.9.6" \
    "safetensors<=0.5.3" \
    peft accelerate \
    scipy tensorboard fire jieba rouge-chinese nltk \
    sse-starlette fastapi uvicorn omegaconf optimum

RUN sed -i 's/d.metadata.get("Name", None)/d.metadata.get("Name", None) if d.metadata else None/g' /usr/local/lib/python3.12/dist-packages/cupy/_environment.py

# 7. 环境设置
WORKDIR /workspace
ENV PYTORCH_ROCM_ARCH="gfx906"
ENV HIP_VISIBLE_DEVICES="0"
ENV HF_HOME="/data/models"
ENV HF_ENDPOINT="https://hf-mirror.com"
ENV RAY_Address="auto"
ENV RAY_USE_MULTIPROCESSING_CPU_COUNT=1
# 禁用版本检查 (防止 LLaMA-Factory 运行时报错说 Pydantic 版本太高)
ENV DISABLE_VERSION_CHECK=1

EXPOSE 8888 8265 8000 7860
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='QazWsx_520'", "--notebook-dir=/workspace"]
