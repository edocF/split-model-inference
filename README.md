## split-model-inference

研究将语音识别模型拆分在边缘设备（移动端）与云端（服务端）进行推理的收益与方法。项目基于 PyTorch 与 torchaudio，使用 LibriSpeech 数据集，支持两种中间特征传输压缩方式：Huffman 编码与自编码器（AutoEncoder）。

参考与源码出处：[`rawahars/split-model-inference`](https://github.com/rawahars/split-model-inference)

### 功能概览
- **完整模型训练/评估**：在单机上训练并评估端到端语音识别模型（CTC）。
- **拆分推理（Split Inference）**：将模型分为 `Head`（边缘端）与 `Tail`（服务端）。边缘端计算中间激活，压缩后通过 TCP 发送至服务端，服务端解压并完成剩余推理与评估。
- **两种编码器**：
  - `huffman`：使用 Huffman 编码压缩中间激活。
  - `autoencoder`：使用训练好的自编码器压缩/解压中间激活。
- **计时与度量**：内置 `Timer` 记录端到端、网络、压缩/解压、推理阶段的时间戳，便于比较不同方案的性能。

### 目录结构
- `data_processing/`：数据预处理（MelSpectrogram）、文本标签转换与贪心解码。
- `model/`：
  - `base_model.py`：端到端 ASR 模型。
  - `split_model.py`：拆分模型的 Head/Tail 定义。
  - `AutoEncoderDecoder.py`：压缩用自编码器结构。
  - `save_load_model.py`：模型加载/保存（含拆分模型与自编码器）。
- `encoder/`：`huffman.py` 与 `AutoEncoders.py` 封装统一编码器接口。
- `distributed_setup/`：基于 `socket` 的客户端/服务端数据传输。
- `train_and_test/`：训练与测试（含拆分场景下的节点脚本）。
- `utilities/Timer.py`：计时工具。
- `main.py`：统一入口（参数化训练/评估/拆分推理）。

### 环境依赖
- Python 3.7+（建议）
- PyTorch / torchaudio（版本见 `setup.py`：`torch==1.4.0`, `torchaudio==0.4.0`）
- 其他：`dahuffman`, `numpy`

安装依赖方式一（自动脚本）：
```bash
python setup.py
```

安装依赖方式二（pip）：
```bash
pip install torch==1.4.0 torchaudio==0.4.0 dahuffman numpy
```

注意：老版本 torch/torchaudio 与系统/驱动兼容性可能受限，必要时请根据本机 CUDA/CPU 情况调整版本。

Windows 读取 FLAC 提示 backend 错误时：
- 已默认在 `main.py` 使用 `torchaudio.set_audio_backend("soundfile")`；若仍报错，请确保安装 `soundfile`（`pip install soundfile`）且系统可用。

### 数据集
首次运行会自动下载 LibriSpeech：
- 训练集：`train-clean-100`
- 测试集：`test-clean`
默认保存在 `./dataset`，可通过参数 `-path` 指定根目录。

### 快速开始
1) 训练完整模型（端到端）并保存参数：
```bash
python main.py -path . -batch 10 -epochs 10 -savefile model.pth
```

2) 使用完整模型进行评估：
```bash
python main.py -test -savefile model.pth -split_mode False -rank 0 -host 127.0.0.1 -port 60009
```

### 拆分推理（Split Inference）
拆分推理需要同时启动两个进程/节点：
- 边缘端（移动端，Rank 0）：运行 `test_head` 逻辑，计算中间层激活并压缩后发送。
- 服务端（Rank 1）：运行 `test_tail` 逻辑，从网络接收压缩数据，解压并完成后续推理与评估。

入口统一使用 `main.py -test -split_mode True`，并通过 `-rank` 区分节点；`-encoder` 指定压缩方式。

#### 方式 A：Huffman 编码（零训练）
1) 在服务端启动（监听）：
```bash
python main.py -test -split_mode True -rank 1 -host 0.0.0.0 -port 60009 -encoder huffman -savefile model.pth
```

2) 在边缘端启动（连接服务端）：
```bash
python main.py -test -split_mode True -rank 0 -host <server_ip_or_hostname> -port 60009 -encoder huffman -savefile model.pth
```

说明：
- 两端需共享同一份权重文件 `model.pth`（用于加载 `Head/Tail` 的结构参数）。
- Huffman 会在边缘端编码时构建 `codec` 并连同形状信息一并发送，服务端据此解码。

#### 方式 B：自编码器（AutoEncoder）压缩
步骤 1：先训练自编码器（基于拆分模型 Head 产生的中间激活进行重建训练）：
```bash
python main.py -path . -batch 10 -epochs 10 -encoder autoencoder -encoderpath autoencoder.pth -savefile model.pth
```

步骤 2：拆分推理（服务端）：
```bash
python main.py -test -split_mode True -rank 1 -host 0.0.0.0 -port 60009 -encoder autoencoder -encoderpath autoencoder.pth -savefile model.pth
```

步骤 3：拆分推理（边缘端）：
```bash
python main.py -test -split_mode True -rank 0 -host <server_ip_or_hostname> -port 60009 -encoder autoencoder -encoderpath autoencoder.pth -savefile model.pth
```

说明：
- `autoencoder.pth` 存储自编码器参数；边缘端用于压缩，服务端用于解压。
- 需要先准备（或训练得到）`model.pth` 以加载拆分模型结构。

### 关键参数（main.py）
- `-test`：是否评估模式；不加则进行训练。
- `-split_mode`：是否启用拆分推理。
- `-rank`：节点角色，`0` 为边缘端、`1` 为服务端（必填且二选一）。
- `-host`：服务端主机名或 IP（边缘端用于连接；服务端可填 `0.0.0.0` 或主机名）。
- `-port`：TCP 端口，默认 `60009`。
- `-path`：数据/模型基路径，默认 `./`。
- `-batch`：批大小，默认 `10`。
- `-epochs`：训练轮数，默认 `10`。
- `-savefile`：完整/拆分模型参数文件名（用于 `load_*`），默认 `model.pth`。
- `-encoder`：`huffman` 或 `autoencoder`。
- `-encoderpath`：自编码器参数文件路径，默认 `autoencoder.pth`。

### 运行细节
- 传输协议：基于 `socket` 的 TCP；客户端（Rank 0）主动连接，服务端（Rank 1）监听并接收。
- 发送数据：`[intermediate, labels, label_lengths, input_lengths, shape, codec]`。
- 计时：`utilities/Timer.py` 记录并可打印各阶段时间戳。
- 解码与评估：服务端执行 CTC 损失、贪心解码（CER/WER）。

### 常见问题
- Torch/torchaudio 版本过旧导致安装失败：根据硬件环境调整版本，或使用 CPU 版。
- 端口占用/防火墙：确保 `-port` 可达，必要时开放端口或更换端口。
- 首次运行下载数据较慢：可提前准备 LibriSpeech 或配代理。

### 许可与引用
本仓库为课程/研究用途示例。如基于此代码进行研究/复现，请引用原仓库：[`rawahars/split-model-inference`](https://github.com/rawahars/split-model-inference)。


"# split-model-inference" 
