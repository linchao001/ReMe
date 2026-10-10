"""LangChain / LlmProvider injection into as_llm for embedded hosts."""

# pylint: disable=protected-access

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agentscope.message import TextBlock, ToolCallBlock, UserMsg
from agentscope.model import ChatModelBase
from agentscope.tool import ToolChoice
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from reme import ReMe
from reme.components.as_llm import (
    LangChainAsLLM,
    LangChainChatModel,
    coerce_to_chat_model,
    is_langchain_compatible,
)
from reme.enumeration import ComponentEnum


class _FakeLangChainChat:
    """Minimal LangChain-shaped chat model."""

    def __init__(self, reply: AIMessage, model_name: str = "fake-lc"):
        self.model_name = model_name
        self.reply = reply
        self.bound_tools = None
        self.bound_tool_choice = None
        self.ainvoke = AsyncMock(side_effect=self._ainvoke)

    async def _ainvoke(self, messages, **_kwargs):
        self.last_messages = messages
        return self.reply

    def bind_tools(self, tools, tool_choice=None):
        self.bound_tools = tools
        self.bound_tool_choice = tool_choice
        return self


class _FakeLlmProvider:
    """AiTester-shaped provider: invoke_messages + bind_tools."""

    def __init__(self, reply: AIMessage, name: str = "provider", model_ref: str = "provider/m"):
        self.name = name
        self.model_ref = model_ref
        self.reply = reply
        self.bound_tools = None
        self.calls = 0

    def bind_tools(self, tools):
        clone = _FakeLlmProvider(self.reply, name=self.name, model_ref=self.model_ref)
        clone.bound_tools = tools
        return clone

    def invoke_messages(self, messages):
        self.calls += 1
        self.last_messages = messages
        return self.reply

    def stream_messages(self, messages):
        self.last_messages = messages
        yield AIMessageChunk(content="hel")
        yield AIMessageChunk(content="lo")


def _qwenpaw_style_config(workspace_dir: str) -> dict:
    return {
        "workspace_dir": workspace_dir,
        "enable_logo": False,
        "log_to_console": False,
        "log_to_file": False,
        "service": {"backend": "http"},
        "jobs": {
            "version": {
                "backend": "base",
                "description": "return reme package version",
                "parameters": {"type": "object", "properties": {}},
                "steps": [{"backend": "version_step"}],
            },
        },
        "components": {
            "as_llm": {
                "default": {
                    "backend": "openai",
                    "model": "consumer-injected",
                    "stream": False,
                    "credential": {"api_key": "", "base_url": ""},
                },
            },
            "agent_wrapper": {
                "default": {
                    "backend": "agentscope",
                    "as_llm": "default",
                },
            },
        },
    }


def test_is_langchain_compatible_detects_provider_and_chat_model():
    assert is_langchain_compatible(_FakeLangChainChat(AIMessage(content="x")))
    assert is_langchain_compatible(_FakeLlmProvider(AIMessage(content="x")))
    assert not is_langchain_compatible(object())
    assert not is_langchain_compatible(None)


def test_coerce_wraps_only_langchain_compatible():
    plain = object()
    assert coerce_to_chat_model(plain) is plain
    wrapped = coerce_to_chat_model(_FakeLangChainChat(AIMessage(content="hi")))
    assert isinstance(wrapped, LangChainChatModel)
    assert isinstance(wrapped, ChatModelBase)


def test_update_component_wraps_langchain_chat_model(tmp_path):
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    fake = _FakeLangChainChat(AIMessage(content="ok", tool_calls=[{"id": "1", "name": "daily_write", "args": {"x": 1}}]))

    async def go():
        component = await app.update_component("as_llm", "default", model=fake)
        assert isinstance(component.model, LangChainChatModel)
        assert component.model.chat_model is fake
        response = await component.model([UserMsg(name="user", content="hi")])
        assert response.is_last is True
        texts = [b.text for b in response.content if isinstance(b, TextBlock)]
        assert texts == ["ok"]
        tools = [b for b in response.content if isinstance(b, ToolCallBlock)]
        assert len(tools) == 1
        assert tools[0].name == "daily_write"
        assert fake.ainvoke.await_count == 1

    asyncio.run(go())


def test_update_component_wraps_aitester_provider(tmp_path):
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    provider = _FakeLlmProvider(AIMessage(content="from-provider"))

    async def go():
        component = await app.update_component("as_llm", "default", model=provider)
        model = component.model
        assert isinstance(model, LangChainChatModel)
        response = await model([UserMsg(name="user", content="hi")])
        assert [b.text for b in response.content if isinstance(b, TextBlock)] == ["from-provider"]
        assert provider.calls == 1
        assert isinstance(provider.last_messages[0], HumanMessage)

    asyncio.run(go())


def test_langchain_chat_model_bind_tools_and_tool_choice():
    fake = _FakeLangChainChat(AIMessage(content="", tool_calls=[{"id": "c1", "name": "generate_structured_output", "args": {"a": 1}}]))
    model = LangChainChatModel(fake, stream=False)

    async def go():
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "generate_structured_output",
                    "description": "structured",
                    "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}},
                },
            },
        ]
        response = await model(
            [UserMsg(name="user", content="hi")],
            tools=tools,
            tool_choice=ToolChoice(mode="generate_structured_output"),
        )
        assert fake.bound_tools == tools
        assert fake.bound_tool_choice == "generate_structured_output"
        assert any(isinstance(b, ToolCallBlock) and b.name == "generate_structured_output" for b in response.content)

    asyncio.run(go())


def test_langchain_chat_model_stream_provider():
    provider = _FakeLlmProvider(AIMessage(content="unused"))
    model = LangChainChatModel(provider, stream=True)

    async def go():
        chunks = []
        async for chunk in await model([UserMsg(name="user", content="hi")]):
            chunks.append(chunk)
        assert any(not c.is_last for c in chunks)
        assert chunks[-1].is_last is True
        assert "".join(b.text for c in chunks for b in c.content if isinstance(b, TextBlock) and not c.is_last) == "hello"

    asyncio.run(go())


def test_langchain_backend_builds_chat_openai(monkeypatch):
    constructed = {}

    class FakeChatOpenAI(_FakeLangChainChat):
        def __init__(self, **kwargs):
            constructed.update(kwargs)
            super().__init__(AIMessage(content="built"), model_name=kwargs.get("model", "m"))

    import sys

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(ChatOpenAI=FakeChatOpenAI))

    llm = LangChainAsLLM(
        name="default",
        backend="langchain",
        model="demo-model",
        stream=False,
        credential={"api_key": "sk-test", "base_url": "https://example.com/v1"},
        parameters={"temperature": 0.2, "max_tokens": 128},
    )
    model = llm.model
    assert isinstance(model, LangChainChatModel)
    assert constructed["model"] == "demo-model"
    assert constructed["api_key"] == "sk-test"
    assert constructed["base_url"] == "https://example.com/v1"
    assert constructed["temperature"] == 0.2
    assert constructed["max_tokens"] == 128


def test_plain_object_injection_still_unwrapped(tmp_path):
    """Compatibility: non-LangChain injections remain verbatim for host tests."""
    app = ReMe(**_qwenpaw_style_config(str(tmp_path)))
    injected = object()

    async def go():
        component = await app.update_component("as_llm", "default", model=injected)
        assert component.model is injected
        assert app.context.components[ComponentEnum.AS_LLM]["default"].model is injected

    asyncio.run(go())
