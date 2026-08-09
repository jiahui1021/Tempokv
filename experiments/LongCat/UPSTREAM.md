# LongCat 上游说明

`longcat_video/` 基于 LongCat-Video 官方仓库提交
`6b3f4b8582a8bc3f20f795735f5383716c4ba794`，上游采用 MIT License。

本仓库只增加 TempoKV 所需的 clean cache、压缩 cache 和解码挂接点。LongCat
模型结构、VAE、scheduler、Context Parallel 与基础生成流程保持上游实现。

上游地址：<https://github.com/meituan-longcat/LongCat-Video>
