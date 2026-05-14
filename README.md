# LLFaigc

ComfyUI 自定义节点库，用于调用 RunningHub OpenAPI。

本库直接保留并复用 `HM-RunningHub/ComfyUI_RH_OpenAPI` 的节点运行逻辑，包括：

- ComfyUI 原生 `IMAGE`、`VIDEO`、`AUDIO` 输入
- 自动上传图片、视频、音频到 RunningHub
- 提交任务、轮询结果、下载图片/视频结果
- `RH_OPENAPI_CONFIG` 配置输入

区别是：`models_registry.json` 只筛选了你需要的图像和视频模型系列，并把目录改为 `LLFaigc-图像生成`、`LLFaigc-视频生成`。

## 模型目录

| 目录 | 系列 | 节点数 |
| --- | --- | ---: |
| LLFaigc-图像生成 | 全能图片G系列 | 6 |
| LLFaigc-图像生成 | 全能图片系列 | 14 |
| LLFaigc-图像生成 | 全能图片X | 8 |
| LLFaigc-视频生成 | 全能视频S系列 | 10 |
| LLFaigc-视频生成 | 可灵系列 | 40 |
| LLFaigc-视频生成 | Wan Video Models系列 | 18 |
| LLFaigc-视频生成 | 全能视频X系列 | 7 |
| LLFaigc-视频生成 | Seedance2.0系列 | 12 |

合计 115 个模型节点。另有 3 个 Seedance2.0 素材辅助节点和 1 个 `LLFaigc-设置` 节点。

## 安装

放到 ComfyUI 的 `custom_nodes` 目录：

```bash
git clone https://github.com/llf-11/LLFaigc.git
```

安装依赖：

```cd LLFaigc
pip install -r requirements.txt
```

然后重启 ComfyUI。

## API Key

可以使用 `LLFaigc OpenAPI 设置` 节点填写 `apiKey`，也可以在环境变量或 `config/.env` 中设置：

```text
RH_API_BASE_URL=https://www.runninghub.cn/openapi/v2
RH_API_KEY=你的 RunningHub API Key
```

## 参考

- API 文档：https://www.runninghub.cn/call-api/api-detail/2004544343584849921?activeTab=api
- 模型价格：https://www.runninghub.cn/third-party-fees
- 参考实现：https://github.com/HM-RunningHub/ComfyUI_RH_OpenAPI
