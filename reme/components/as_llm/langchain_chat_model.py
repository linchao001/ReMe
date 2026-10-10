"""AgentScope ChatModelBase adapter for LangChain chat models and providers.

Embedded hosts (for example AiTester) often own a LangChain ``ChatOpenAI`` or a
provider with ``invoke_messages`` / ``bind_tools``. ``AsAgentWrapper`` still
requires ``agentscope.model.ChatModelBase``. This adapter bridges the two so
hosts can inject their live chat object into ``as_llm`` without constructing a
second AgentScope provider from the same credentials.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncGenerator, Iterator
from datetime import datetime
from typing import Any

from agentscope.credential import OpenAICredential
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg, TextBlock, ToolCallBlock
from agentscope.model import ChatModelBase, ChatResponse, ChatUsage
from agentscope.tool import ToolChoice
from pydantic import BaseModel, Field


def _is_real_callable(value: Any, name: str) -> bool:
    """Return True for a real callable attribute, never for ``unittest.mock`` stand-ins."""
    attr = getattr(value, name, None)
    if not callable(attr):
        return False
    module = getattr(type(attr), "__module__", "") or ""
    return not module.startswith("unittest.mock")


def is_langchain_compatible(value: Any) -> bool:
    """Return True when *value* looks like a LangChain chat model or LlmProvider."""
    if value is None or isinstance(value, ChatModelBase):
        return False
    module = getattr(type(value), "__module__", "") or ""
    # Bare mocks expose every attribute as a callable; never auto-wrap them.
    if module.startswith("unittest.mock"):
        return False
    if _is_real_callable(value, "invoke_messages") and _is_real_callable(value, "bind_tools"):
        return True
    has_invoke = callable(getattr(value, "ainvoke", None)) or callable(getattr(value, "invoke", None))
    if module.startswith("langchain") and has_invoke:
        return True
    # Duck-typed chat models (including test doubles with AsyncMock ainvoke).
    if _is_real_callable(value, "bind_tools") and has_invoke:
        return True
    return False


def coerce_to_chat_model(
    value: Any,
    *,
    stream: bool = False,
    context_size: int = 200000,
    max_retries: int = 3,
    model_name: str | None = None,
) -> Any:
    """Wrap LangChain-compatible objects as ``ChatModelBase``; leave others untouched."""
    if not is_langchain_compatible(value):
        return value
    return LangChainChatModel(
        value,
        model=model_name,
        stream=stream,
        context_size=context_size,
        max_retries=max_retries,
    )


def _model_name_of(chat_model: Any, explicit: str | None) -> str:
    if explicit:
        return explicit
    for attr in ("model_name", "model", "model_ref", "name"):
        candidate = getattr(chat_model, attr, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return "langchain"


def _to_langchain_messages(openai_messages: list[dict[str, Any]]) -> list[Any]:
    """Convert OpenAI-format dicts to LangChain messages when the helper exists."""
    try:
        from langchain_core.messages import convert_to_messages
    except ImportError:
        return openai_messages
    return convert_to_messages(openai_messages)


def _tool_choice_for_langchain(tool_choice: ToolChoice | None) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    mode = tool_choice.mode
    if mode in {"auto", "none", "required"}:
        return mode
    return mode


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "".join(parts)
    return str(content)


def _tool_call_input(args: Any) -> str:
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args if args is not None else {}, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"value": str(args)}, ensure_ascii=False)


def _usage_from_message(message: Any, started: datetime) -> ChatUsage | None:
    usage_metadata = getattr(message, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        input_tokens = int(usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0)
        output_tokens = int(usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0)
        if input_tokens or output_tokens:
            return ChatUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                time=(datetime.now() - started).total_seconds(),
            )
    metadata = getattr(message, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
    if isinstance(token_usage, dict):
        input_tokens = int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0)
        output_tokens = int(token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0)
        if input_tokens or output_tokens:
            return ChatUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                time=(datetime.now() - started).total_seconds(),
            )
    return None


def _message_to_chat_response(message: Any, started: datetime) -> ChatResponse:
    content: list[TextBlock | ToolCallBlock] = []
    text = _content_to_text(getattr(message, "content", message))
    if text:
        content.append(TextBlock(text=text))

    tool_calls = getattr(message, "tool_calls", None) or []
    for call in tool_calls:
        if isinstance(call, dict):
            call_id = str(call.get("id") or "")
            name = str(call.get("name") or "")
            args = call.get("args", call.get("arguments", {}))
        else:
            call_id = str(getattr(call, "id", "") or "")
            name = str(getattr(call, "name", "") or "")
            args = getattr(call, "args", None)
            if args is None:
                args = getattr(call, "arguments", {})
        if not name:
            continue
        content.append(ToolCallBlock(id=call_id or name, name=name, input=_tool_call_input(args)))

    if not content:
        content.append(TextBlock(text=""))

    return ChatResponse(
        content=content,
        is_last=True,
        usage=_usage_from_message(message, started),
    )


class LangChainChatModel(ChatModelBase):
    """``ChatModelBase`` that delegates generation to a LangChain-compatible object."""

    class Parameters(BaseModel):
        """Optional generation knobs forwarded only when constructing from config."""

        temperature: float | None = Field(default=None)
        max_tokens: int | None = Field(default=None)

    def __init__(
        self,
        chat_model: Any,
        *,
        model: str | None = None,
        parameters: Parameters | None = None,
        stream: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        context_size: int = 200000,
        formatter: OpenAIChatFormatter | None = None,
    ) -> None:
        super().__init__(
            credential=OpenAICredential(api_key="langchain"),
            model=_model_name_of(chat_model, model),
            parameters=parameters or self.Parameters(),
            stream=stream,
            max_retries=max_retries,
            retry_delay=retry_delay,
            context_size=context_size,
        )
        self.chat_model = chat_model
        self.formatter = formatter or OpenAIChatFormatter()

    def _bind_tools(self, tools: list[dict] | None, tool_choice: ToolChoice | None) -> Any:
        runnable = self.chat_model
        if not tools:
            return runnable
        bind = getattr(runnable, "bind_tools", None)
        if not callable(bind):
            raise TypeError(f"{type(runnable).__name__} does not support bind_tools required by Agent tools")
        kwargs: dict[str, Any] = {}
        lc_choice = _tool_choice_for_langchain(tool_choice)
        if lc_choice is not None:
            params = inspect.signature(bind).parameters
            if "tool_choice" in params:
                kwargs["tool_choice"] = lc_choice
        return bind(tools, **kwargs)

    async def _invoke(self, runnable: Any, messages: list[Any]) -> Any:
        if callable(getattr(runnable, "ainvoke", None)):
            return await runnable.ainvoke(messages)
        if callable(getattr(runnable, "invoke_messages", None)):
            return await asyncio.to_thread(runnable.invoke_messages, messages)
        if callable(getattr(runnable, "invoke", None)):
            return await asyncio.to_thread(runnable.invoke, messages)
        raise TypeError(
            f"{type(runnable).__name__} is not LangChain-compatible "
            "(expected ainvoke/invoke or invoke_messages)",
        )

    async def _iter_stream_chunks(self, runnable: Any, messages: list[Any]) -> AsyncGenerator[Any, None]:
        if callable(getattr(runnable, "astream", None)):
            async for chunk in runnable.astream(messages):
                yield chunk
            return
        stream_messages = getattr(runnable, "stream_messages", None)
        if callable(stream_messages):

            def _collect() -> list[Any]:
                return list(stream_messages(messages))

            for chunk in await asyncio.to_thread(_collect):
                yield chunk
            return
        stream = getattr(runnable, "stream", None)
        if callable(stream):

            def _collect_stream() -> list[Any]:
                result = stream(messages)
                return list(result) if isinstance(result, Iterator) else list(result)

            for chunk in await asyncio.to_thread(_collect_stream):
                yield chunk
            return
        yield await self._invoke(runnable, messages)

    async def _call_api(
        self,
        model_name: str,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        del model_name, kwargs
        self._validate_tool_choice(tool_choice, tools)
        openai_messages = await self.formatter.format(messages)
        lc_messages = _to_langchain_messages(openai_messages)
        runnable = self._bind_tools(tools, tool_choice)
        started = datetime.now()

        if self.stream:
            return self._stream_response(runnable, lc_messages, started)
        message = await self._invoke(runnable, lc_messages)
        return _message_to_chat_response(message, started)

    async def _stream_response(
        self,
        runnable: Any,
        messages: list[Any],
        started: datetime,
    ) -> AsyncGenerator[ChatResponse, None]:
        """Yield text deltas when available, then one assembled ``is_last`` response."""
        text_parts: list[str] = []
        last_message: Any = None
        async for chunk in self._iter_stream_chunks(runnable, messages):
            last_message = chunk
            delta_text = _content_to_text(getattr(chunk, "content", None))
            if delta_text:
                text_parts.append(delta_text)
                yield ChatResponse(content=[TextBlock(text=delta_text)], is_last=False)

        if last_message is None:
            yield ChatResponse(content=[TextBlock(text="")], is_last=True)
            return

        # Prefer the provider's final message (includes tool_calls) when present.
        if getattr(last_message, "tool_calls", None) or not text_parts:
            yield _message_to_chat_response(last_message, started)
            return

        yield ChatResponse(
            content=[TextBlock(text="".join(text_parts))],
            is_last=True,
            usage=_usage_from_message(last_message, started),
        )


__all__ = [
    "LangChainChatModel",
    "coerce_to_chat_model",
    "is_langchain_compatible",
]
