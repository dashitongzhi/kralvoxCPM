# 声创科技方言语音生成部署说明

这个分支在原 VoxCPM Gradio 页面上增加了一条产品链路：

普通话文本 + 目标方言 + 角色画像/声音描述 -> 方言文本改写 -> VoxCPM 语音生成

## 运行前准备

1. 克隆仓库并安装依赖。

```bash
git clone https://github.com/dashitongzhi/kralvoxCPM.git
cd kralvoxCPM
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

如果机器已有 conda，也可以使用 conda 环境；重点是不要使用系统 Python 直接安装大依赖。

2. 下载或放置预训练模型。

推荐把模型放到数据盘，例如：

```text
/root/autodl-tmp/models/VoxCPM2
```

目录内至少需要包含：

```text
config.json
model.safetensors
audiovae.pth
tokenizer.json
tokenizer_config.json
special_tokens_map.json
tokenization_voxcpm2.py
```

3. 配置方言改写接口和本地模型路径。

复制模板并填写真实密钥：

```bash
cp .env.example .env
```

至少需要配置：

```bash
KRALAPI_BASE_URL=https://kralapi.kralai.tech
KRALAPI_MODEL=gpt-5.5
KRALAPI_API_KEY=你的接口密钥
VOXCPM_MODEL_PATH=/root/autodl-tmp/models/VoxCPM2
HOST=0.0.0.0
PORT=8808
DEVICE=cuda
```

不要把真实 `.env`、`.runtime.env` 或 API key 提交到仓库。

## 启动

推荐使用仓库内的最小启动脚本。脚本会在启动前检查 `.env`、`KRALAPI_API_KEY`、`VOXCPM_MODEL_PATH`、模型关键文件、Python 依赖、CUDA 可用性和 `8808` 端口占用；如果缺配置或路径不对，会直接给出明确报错。

```bash
chmod +x scripts/run_dialect_demo.sh
scripts/run_dialect_demo.sh
```

浏览器打开：

```text
http://127.0.0.1:8808
```

远程服务器建议通过 SSH 隧道访问：

```bash
ssh -p 17591 -L 8808:127.0.0.1:8808 root@connect.bjb1.seetacloud.com
```

如果需要临时改端口或设备，不用改代码：

```bash
PORT=8810 DEVICE=auto scripts/run_dialect_demo.sh
```

也可以不用脚本，手动启动：

```bash
set -a
source .env
set +a
python app.py \
  --model-id "$VOXCPM_MODEL_PATH" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8808}" \
  --device "${DEVICE:-cuda}"
```

## 使用方式

页面里填写：

- 目标方言：例如 `粤语`
- 角色画像/声音描述：例如 `暴躁的广东中年男教练，语速快，声音粗粝，充满无奈和愤怒`
- 普通话文本：用户想要生成的普通话原文

点击“生成语音”后，页面会先显示“已改写的方言文本”，然后生成音频。

## 数据集和输出目录

本仓库不提交大模型、数据集、缓存和生成音频。推荐目录：

```text
models/       # 本机预训练模型，可自建
datasets/     # 本机训练或测试数据，可自建
outputs/      # 生成音频，可自建
cache/        # Hugging Face / ModelScope / Torch 缓存，可自建
```

这些目录已在 `.gitignore` 中忽略。换机器时，把仓库克隆下来，再把预训练模型和数据集放到对应目录或通过 `--model-id` 指向实际路径即可。
