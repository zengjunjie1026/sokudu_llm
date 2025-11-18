"""
统一的 LLM Chat Completion 客户端，支持多个 OpenAI 兼容的供应商。

目前内置支持：
- openai: 使用 https://api.openai.com/v1/chat/completions
- deepseek: 使用 https://api.deepseek.com/v1/chat/completions
- qwen (DashScope 兼容模式): https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions

如果需要扩展更多供应商，可在 PROVIDERS 常量中追加配置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from loguru import logger


class LLMClientError(RuntimeError):
    """在调用远程 LLM API 时出现的错误。"""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str
    default_model: str
    request_timeout: float = 120.0

    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-5",
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
    ),
    "qwen": ProviderConfig(
        name="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen-max-latest",
    ),
}


def get_provider(provider: str) -> ProviderConfig:
    try:
        return PROVIDERS[provider]
    except KeyError as exc:
        raise ValueError(f"不支持的 provider: {provider}") from exc


def ensure_api_key(config: ProviderConfig) -> str:
    key = os.getenv(config.api_key_env)
    if not key:
        raise EnvironmentError(
            f"未检测到 {config.api_key_env} 环境变量，请先设置 API Key 再运行脚本。"
        )
    return key


def normalize_reasoning(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        for key in ("text", "content"):
            maybe = value.get(key)
            if isinstance(maybe, str) and maybe.strip():
                return maybe.strip()
        if "tokens" in value and isinstance(value["tokens"], list):
            combined = "".join(
                token.get("text", "")
                for token in value["tokens"]
                if isinstance(token, dict)
            )
            combined = combined.strip()
            return combined or None
        return None
    if isinstance(value, list):
        combined = "".join(
            item for item in (normalize_reasoning(v) or "" for v in value)
        ).strip()
        return combined or None
    return str(value).strip() or None


def chat_completion(
    provider: str,
    messages: List[Dict[str, str]],
    temperature: float = 1.0,
    model: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    调用指定 provider 的聊天补全接口。

    Args:
        provider: 供应商名称（openai / deepseek / qwen）
        messages: 聊天消息列表
        temperature: 温度参数
        model: 使用的模型名称；若为空则使用 provider 默认模型

    Returns:
        (content, reasoning) 二元组。reasoning 可能为 None。
    """

    config = get_provider(provider)
    api_key = ensure_api_key(config)
    payload = {
        "model": model or config.default_model,
        "messages": messages,
        "temperature": temperature,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    endpoint = config.endpoint()
    logger.debug(
        "POST {endpoint} | provider={provider} model={model} temperature={temperature} messages={count}",
        endpoint=endpoint,
        provider=config.name,
        model=payload["model"],
        temperature=temperature,
        count=len(messages),
    )

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=config.request_timeout,
        )
    except requests.RequestException as exc:  # pragma: no cover - 运行期错误
        raise LLMClientError(f"调用 {config.name} 接口失败: {exc}") from exc

    if response.status_code >= 400:
        raise LLMClientError(
            f"{config.name} 接口返回错误 {response.status_code}: {response.text}"
        )

    data = response.json()
    try:
        choice = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMClientError(f"{config.name} 返回数据格式异常: {data}") from exc

    content = (choice.get("content") or "").strip()
    reasoning = normalize_reasoning(choice.get("reasoning"))

    logger.debug(
        "Response {endpoint} | provider={provider} status={status} content_preview={preview}",
        endpoint=endpoint,
        provider=config.name,
        status=response.status_code,
        preview=(content.splitlines()[0][:120] if content else ""),
    )

    return content, reasoning


__all__ = [
    "chat_completion",
    "get_provider",
    "PROVIDERS",
    "LLMClientError",
]

