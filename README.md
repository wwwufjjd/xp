# 多参考媒体 XP 特征筛选

这是一个本地 Gradio 应用：先用参考图片/视频封面生成可编辑的 XP 特征，再给本地图片/视频媒体库生成封面特征缓存，最后用文字特征匹配筛选。在线视频封面可单独采集，只保存封面图和原链接，不下载视频。

## 模型

- 默认：`Disty0/Qwen3-VL-8B-NSFW-Caption-V4.5`，用于更细的人物和身材特征描述。
- 可选：`Camais03/camie-tagger-v2`，用于更快的批量标签扫描。
- Qwen 默认使用 4bit 加载，适合 V100 16GB；可用 `XP_QWEN_LOAD_IN_4BIT=0` 改为 FP16。
- Qwen 默认把输入图限制到 `512*512` 像素，避免高分辨率图 OOM；可用 `XP_QWEN_MAX_PIXELS` 调整。

首次生成特征会从 Hugging Face 下载权重，缓存到 `models/hf-cache/`。

## 安装

```powershell
cd D:\ai\h
python -m pip install -r requirements.txt
```

本地视频封面依赖 `ffmpeg/ffprobe`。程序会优先读取环境变量 `XP_FFMPEG` / `XP_FFPROBE`，否则尝试系统 PATH 和 `D:\1\ffmpeg-6.1.1-essentials_build\bin\`。

## 启动

```powershell
cd D:\ai\h
python app.py
```

打开：

```text
http://127.0.0.1:7860
```

## 使用流程

1. 在“生成 XP 特征”里上传参考图片/视频，或输入参考媒体文件夹路径。
2. 在 XP 特征文本框里删除不想要的词，或手动改写。
3. 在“生成/更新本地媒体库特征”里输入图片/视频文件夹路径并生成缓存。
4. 可选：在“在线视频封面采集”里按站点采集封面并生成特征。
5. 在“匹配筛选”里选择参与筛选的字段，点击“用 XP 特征匹配媒体库”。

## 在线封面

- `Pornhub`：输入关键词，采集公开搜索页能访问到的标题、链接和封面。
- `通用网页/oEmbed/OpenGraph`：输入视频页 URL，读取 oEmbed 或 OpenGraph 封面。
- `Telegram`：使用 Telethon 用户账号 session，读取用户账号可访问的频道/聊天视频消息封面。

缓存文件写到 `data/library.db`，缩略图写到 `data/thumbs/`，本地视频封面写到 `data/covers/`，在线封面写到 `data/online_covers/`。缓存会按模型、prompt 版本、粗/细模式和缓存档区分。
