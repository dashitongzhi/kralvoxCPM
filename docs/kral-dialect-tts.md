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

2. 准备远程数据盘和预训练模型。

推荐统一把远程云盘/数据盘挂到 `DATA_ROOT`，例如：

```text
DATA_ROOT=/root/autodl-tmp
```

8808 demo 默认使用这个目录结构：

```text
$DATA_ROOT/
  models/
    VoxCPM2/
  cache/
    hf/
    modelscope/
    torch/
```

其中模型目录默认是 `$DATA_ROOT/models/VoxCPM2`，目录内至少需要包含：

```text
config.json
model.safetensors（或其他 *.safetensors/*.bin 主权重文件）
audiovae.pth
tokenizer.json
tokenizer_config.json
special_tokens_map.json
tokenization_voxcpm2.py
```

3. 运行 8808 demo 的一键准备脚本。

```bash
DATA_ROOT=/root/autodl-tmp scripts/prepare_dialect_demo.sh
```

脚本会做三件事：

- 如果 `.env` 不存在，就按 `.env.example` 生成一份可修改的本地模板；如果已经存在，则只读取检查，不覆盖。
- 确认 `DATA_ROOT` 挂载点已存在，并创建 8808 demo 需要的 `$DATA_ROOT/models/VoxCPM2`、`$DATA_ROOT/cache/hf`、`$DATA_ROOT/cache/modelscope` 和 `$DATA_ROOT/cache/torch`。
- 提示还需要手动补齐的配置或模型文件。

如果已经在 `.env` 里写好 `DATA_ROOT`，也可以直接运行：

```bash
scripts/prepare_dialect_demo.sh
```

4. 填写方言改写接口密钥，并放好预训练模型。

编辑 `.env`，至少确认这些配置：

```bash
KRALAPI_BASE_URL=https://kralapi.kralai.tech
KRALAPI_MODEL=gpt-5.5
KRALAPI_API_KEY=你的接口密钥
DATA_ROOT=/root/autodl-tmp
HOST=0.0.0.0
PORT=8808
DEVICE=cuda
```

设置 `DATA_ROOT` 后，启动脚本会默认派生：

```bash
VOXCPM_MODEL_PATH=$DATA_ROOT/models/VoxCPM2
HF_HOME=$DATA_ROOT/cache/hf
MODELSCOPE_CACHE=$DATA_ROOT/cache/modelscope
TORCH_HOME=$DATA_ROOT/cache/torch
```

如果模型或缓存确实放在别处，可以在 `.env` 里单独覆盖这些变量。真实模型文件仍然需要手动放到 `$DATA_ROOT/models/VoxCPM2`，或者把 `VOXCPM_MODEL_PATH` 指向实际模型目录。

不要把真实 `.env`、`.runtime.env` 或 API key 提交到仓库。

## 启动

推荐使用仓库内的最小启动脚本。准备阶段先运行 `scripts/prepare_dialect_demo.sh`；启动脚本会在启动前检查 `.env`、`KRALAPI_API_KEY`、`DATA_ROOT` 挂载目录、`models`/`cache` 子目录、`VOXCPM_MODEL_PATH`、模型关键文件、Python 依赖、CUDA 可用性和 `8808` 端口占用；如果缺配置或路径不对，会直接给出明确报错。

```bash
scripts/prepare_dialect_demo.sh
scripts/run_dialect_demo.sh
```

浏览器打开：

```text
http://127.0.0.1:8808
```

页面顶部的“服务就绪状态”会按项提示远端 8808 demo 的缺口：`DATA_ROOT 挂载`异常表示数据盘没挂载或变量写错，`models/cache 目录`异常表示挂载点能访问但缺少 demo 子目录，`模型路径`异常表示 VoxCPM2 模型目录不存在，或缺少关键模型文件/主权重文件。

远程服务器建议通过 SSH 隧道访问：

```bash
ssh -p 17591 -L 8808:127.0.0.1:8808 root@connect.bjb1.seetacloud.com
```

如果需要临时改端口或设备，不用改代码：

```bash
PORT=8810 DEVICE=auto scripts/run_dialect_demo.sh
```

常见数据盘报错含义：

- `DATA_ROOT mount directory does not exist`：云盘/数据盘没有挂载好，或 `DATA_ROOT` 写错了。
- `DATA_ROOT is reachable, but its required demo subdirectories are missing`：挂载目录能访问，但还没准备 `models/`、`cache/` 目录。
- `Model directory is missing under DATA_ROOT` 或 `Model directory is incomplete`：云盘挂载是通的，但 VoxCPM2 模型目录不存在或模型文件不全。

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
$DATA_ROOT/models/       # 远程数据盘上的预训练模型
$DATA_ROOT/datasets/     # 训练或测试数据，可自建
$DATA_ROOT/outputs/      # 生成音频，可自建
$DATA_ROOT/cache/        # Hugging Face / ModelScope / Torch 缓存
```

这些目录已在 `.gitignore` 中忽略。换机器时，把仓库克隆下来，再把预训练模型和数据集放到对应目录或通过 `--model-id` 指向实际路径即可。
