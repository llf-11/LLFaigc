# LLFaigc

ComfyUI 自定义节点库，用于调用 RunningHub API。

## 支持的模型系列

| 类型 | 模型系列 | 价格 |
| --- | --- | --- |
| 图像生成 | 全能图片G系列 | 以 RunningHub 模型API价格页为准；公开索引示例：全能图片G-1.5-文生图 4.7 w |
| 图像生成 | 全能图片系列 | 以 RunningHub 模型API价格页为准 |
| 图像生成 | 全能图片X | 以 RunningHub 模型API价格页为准 |
| 视频生成 | 全能视频S系列 | 以 RunningHub 模型API价格页为准；公开索引示例：全能视频S 9.8 w |
| 视频生成 | 可灵系列 | 以 RunningHub 模型API价格页为准；公开索引示例：可灵文生视频o3-pro 38.5 w |
| 视频生成 | Wan Video Models系列 | 以 RunningHub 模型API价格页为准 |
| 视频生成 | 全能视频X系列 | 以 RunningHub 模型API价格页为准 |
| 视频生成 | Seedance2.0系列 | 以 RunningHub 模型API价格页为准 |

价格页：https://www.runninghub.cn/third-party-fees

API 文档页：https://www.runninghub.cn/call-api/api-detail/2004544343584849921?activeTab=api

> RunningHub 的价格页和 API 详情页是动态页面。节点内置了模型系列和可维护的价格字段，实际扣费请以 RunningHub 后台和价格页实时显示为准。

## 节点分支

本库参考 `HM-RunningHub/ComfyUI_RH_OpenAPI` 的模型注册方式，从其 `models_registry.json` 中筛选了你指定的模型系列，并排除了标记为已下架或 Deprecated 的模型。

当前共注册 115 个模型节点：

| 分支 | 系列 | 节点数 |
| --- | --- | ---: |
| 图像生成 | 全能图片G系列 | 6 |
| 图像生成 | 全能图片系列 | 14 |
| 图像生成 | 全能图片X | 8 |
| 视频生成 | 全能视频S系列 | 10 |
| 视频生成 | 可灵系列 | 40 |
| 视频生成 | Wan Video Models系列 | 18 |
| 视频生成 | 全能视频X系列 | 7 |
| 视频生成 | Seedance2.0系列 | 12 |

在 ComfyUI 中的分类路径示例：

```text
LLFaigc/RunningHub/图像生成/全能图片G系列
LLFaigc/RunningHub/视频生成/可灵系列
```

## 安装

在 ComfyUI 的 `custom_nodes` 目录中执行：

```bash
git clone https://github.com/llf-11/LLFaigc.git
```

然后重启 ComfyUI。

## 节点

### LLFaigc RunningHub 模型信息

输出模型名称、类型、价格说明、文档链接。

### 各模型独立节点

每个 RunningHub 模型都会生成一个独立节点。输入 RunningHub API Key、prompt、图片 URL、视频 URL 等参数，返回：

- `task_id`
- `urls`
- `response_json`
- `payload_json`

媒体参数目前使用 URL 字符串接入，例如 `imageUrl`、`imageUrls`、`videoUrl`。多个 URL 可以换行填写。

`raw_json` 可以覆盖或追加 RunningHub API 所需的参数。例如：

```json
{
  "workflowId": "your-workflow-id",
  "nodeInfoList": [],
  "webhookUrl": ""
}
```

如果 RunningHub 调整接口路径，可以直接在节点里修改 `create_endpoint` 和 `status_endpoint`。

## 说明

RunningHub 不同模型系列的参数可能不同。这个节点库采用“模型独立节点 + raw JSON 覆盖”的方式，保证在 RunningHub API 参数更新时仍能快速适配。
