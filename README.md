# XP 本地媒体工具

这个仓库包含两个独立程序：

- `app.py`：本地 Gradio 图形界面，用参考图片/视频生成 XP 特征，再匹配本地或在线封面媒体库。
- `vocal_intensity.py`：命令行视频音频分析器，按全片扫描结果给视频排序，并截取音频高峰片段。

运行缓存、模型、数据库、缩略图和截取片段默认都写在本地目录，已经通过 `.gitignore` 排除。

## 安装

```powershell
cd E:\xp-main
python -m pip install -r requirements.txt
```

视频封面和音频分析都需要 `ffmpeg/ffprobe`。程序会优先读取 `XP_FFMPEG` / `XP_FFPROBE`，否则尝试系统 `PATH` 和 `D:\1\ffmpeg-6.1.1-essentials_build\bin\`。

## 程序一：XP 特征筛选界面

### 功能

- 从参考图片或视频封面生成可编辑的 XP 特征文本。
- 扫描本地图片/视频文件夹，生成封面、缩略图和特征缓存。
- 支持采集在线视频页面封面，只保存标题、链接和封面图，不下载视频。
- 支持 Telegram 可访问聊天/频道的视频消息封面采集。
- 按选择字段和权重，用 XP 特征匹配媒体库并排序。

### 启动

```powershell
cd E:\xp-main
python app.py
```

或双击：

```text
start.bat
```

浏览器打开：

```text
http://127.0.0.1:7860
```

### 使用流程

1. 在“生成 XP 特征”里上传参考图片/视频，或输入参考媒体文件夹路径。
2. 检查生成的特征文本，删除不想参与匹配的描述，必要时手动改写。
3. 在“生成/更新本地媒体库特征”里输入本地媒体目录，生成缓存。
4. 可选：在“在线视频封面采集”里采集网页、站点搜索页或 Telegram 封面。
5. 在“匹配筛选”里选择字段和排序方式，点击匹配。

### 常用环境变量

- `XP_APP_PORT`：Gradio 端口，默认 `7860`。
- `XP_MODEL_CACHE_DIR`：Hugging Face 模型缓存目录，默认 `models/hf-cache/`。
- `XP_QWEN_MODEL_ID`：本地图像理解模型，默认 `Disty0/Qwen3-VL-8B-NSFW-Caption-V4.5`。
- `XP_QWEN_LOAD_IN_4BIT`：默认 `1`，设为 `0` 使用 FP16。
- `XP_QWEN_MAX_PIXELS`：输入图片最大像素，默认 `512*512`。
- `XP_ENABLE_OPENAI_BACKEND`：默认 `1`，设为 `0` 隐藏 OpenAI API 后端。
- `XP_FEATURE_BACKEND`：默认特征生成后端。
- `XP_OPENAI_API_KEY`、`XP_OPENAI_BASE_URL`、`XP_OPENAI_MODEL_ID`：OpenAI 兼容接口配置。

不要把 API key 写进仓库文件。`start.bat` 只读取你已经设置好的环境变量。

### 本地数据位置

- `data/library.db`：媒体库缓存数据库。
- `data/thumbs/`：缩略图。
- `data/covers/`：本地视频封面。
- `data/online_covers/`：在线封面。
- `models/hf-cache/`：模型缓存。

## 程序二：视频音频激烈度分析器

### 功能

- 使用 `signal` 引擎，不依赖模型，扫描完整视频音轨。
- 综合绝对音量、相对峰值、人声频带、事件持续时间、对比度和片段稳定性排序。
- 为每个视频选出最强候选高峰段，并可截取 `_clips` 片段。
- 支持 JSON 输出和 HTML 报告。
- 兼容旧版 `--engine` 参数；即使传入 `yamnet`、`clap` 或 `light`，当前版本也会自动改用 `signal`。

### 基本用法

分析一个目录：

```powershell
cd E:\xp-main
python vocal_intensity.py "D:\videos"
```

输出 JSON：

```powershell
python vocal_intensity.py "D:\videos" -o result.json
```

截取每个视频最强的 3 个候选片段：

```powershell
python vocal_intensity.py "D:\videos" --clip --clip-top 3 --clip-sec 10 -o result.json
```

只显示前 20 个：

```powershell
python vocal_intensity.py "D:\videos" -t 20
```

兼容旧命令参数：

```powershell
python vocal_intensity.py "D:\videos" --engine yamnet
```

### 主要参数

- `target`：视频文件或目录。
- `--engine`：兼容旧参数，当前统一使用 `signal`。
- `--clip`：截取高峰片段。
- `--clip-sec`：每段长度，默认 `10` 秒。
- `--clip-top`：每个视频截取前 N 个候选高峰，默认 `1`。
- `--min-score`：低于该分数不截取，默认 `0.3`。
- `-o, --output`：保存 JSON 结果。
- `-t, --top`：只显示前 N 个结果。
- `--sort`：排序字段，可选 `intensity`、`max`、`loud_ratio`、`name`。

### 输出说明

- 终端会显示每个视频的总分、最强时间点、候选区间和关键音频特征。
- 开启 `--clip` 后，片段会写入目标目录下的 `_clips/`。
- 重新测试时建议先删除旧 `_clips/`，避免把之前截取出来的短片也混进待分析目录。
- 分数是排序用的启发式指标，不是绝对真实标签；如果实际观感有偏差，优先比较同一批视频内的相对排名。
