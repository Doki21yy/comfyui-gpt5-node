# ComfyUI GPT-5 Vision + Knowledge Node

精简版 GPT 节点，专门用于 OpenAI GPT（固定 Responses API），支持：

- 系统提示词（system instructions）
- 用户文本 + 多图输入（最多 4 张）
- 知识库检索（上传多个 `.md` 文件后自动检索）

## 设计目标

对齐 GPTs 使用体验：

- 不需要选择 provider
- 不需要选择 chat/responses 模式
- 不需要手动填写 vector_store_id
- 直接上传 `.md`，节点显示文件名并自动生效

## 安装

放到：

```txt
ComfyUI/custom_nodes/comfyui-gpt5-node/
```

安装依赖：

```bash
pip install openai
```

重启 ComfyUI。

## 使用

1. 添加节点：`GPT5 / Chat / GPT-5 Vision + Knowledge`
2. 连接 `api_key`（STRING 输入）
3. 填 `system_content` 和 `prompt`
4. 需要知识库时，点击节点按钮 `Upload .md Files`，选择多个 `.md`
5. 可选连接 `images`（推荐接 `Images Batch Multiple` 输出）

## 输入参数

- `prompt`: 用户输入
- `system_content`: 系统提示词
- `model_name`: 模型名，默认 `gpt-5.3-chat-latest`
- `max_output_tokens`: 最大输出 token
- `reasoning_effort`: 推理强度
- `image_detail`: 图片细节等级
- `knowledge_files`: 上传后自动维护的文件名列表（通常无需手填）
- `images`: 多图批量输入（IMAGE batch）
- `api_key`: OpenAI API Key（必填）

## 输出

- `text`: 模型回答
- `request_debug`: 请求摘要（已缩写）
- `knowledge_info`: 知识库状态、vector_store_id、文件列表
- `finish_reason`: 响应状态
- `citations`: 文件引用信息

## 说明

- API base URL 已硬编码为 `https://api.openai.com/v1`
- 节点仅使用 `Responses API`
- 知识库仅处理 `.md` 文件
