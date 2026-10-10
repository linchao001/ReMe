"""LLM model wrappers for AgentScope."""

from typing import Any

from agentscope.credential import (
    AnthropicCredential,
    CredentialBase,
    DashScopeCredential,
    DeepSeekCredential,
    GeminiCredential,
    MoonshotCredential,
    OllamaCredential,
    OpenAICredential,
    XAICredential,
)
from agentscope.model import ChatModelBase

from .langchain_chat_model import LangChainChatModel, coerce_to_chat_model, is_langchain_compatible
from ..base_component import BaseComponent
from ..component_registry import R
from ...enumeration import ComponentEnum


def _credential_api_key(credential_data: dict) -> str:
    api_key = credential_data.get("api_key", "")
    if hasattr(api_key, "get_secret_value"):
        api_key = api_key.get_secret_value()
    return str(api_key or "").strip()


class BaseAsLLM(BaseComponent):
    """Base wrapper for AgentScope chat models.

    Subclasses set ``credential_cls``. Providers are constructed on first use,
    allowing local file and search jobs to run without model credentials.

    Embedded hosts may inject a live AgentScope ``ChatModelBase``, a LangChain
    chat model, or an AiTester-style provider (``invoke_messages`` /
    ``bind_tools``). LangChain-compatible objects are coerced to
    ``LangChainChatModel`` so ``AsAgentWrapper`` keeps a uniform contract.
    """

    component_type = ComponentEnum.AS_LLM
    credential_cls: type[CredentialBase]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model: ChatModelBase | None = None

    async def _start(self) -> None:
        """Keep service startup independent of provider credentials."""

    def _coerce_injected_model(self, value: Any) -> Any:
        """Wrap LangChain-compatible injections using this component's stream settings."""
        if not is_langchain_compatible(value):
            return value
        return coerce_to_chat_model(
            value,
            stream=bool(self.kwargs.get("stream", False)),
            context_size=int(self.kwargs.get("context_size", 200000)),
            max_retries=int(self.kwargs.get("max_retries", 3)),
            model_name=self.kwargs.get("model") if isinstance(self.kwargs.get("model"), str) else None,
        )

    @property
    def model(self) -> ChatModelBase:
        """Initialize on first access, including direct Step dependency resolution."""
        self.initialize_model()
        assert self._model is not None
        return self._model

    @model.setter
    def model(self, value: ChatModelBase | None) -> None:
        """Preserve explicit model injection used by standalone consumers."""
        self._model = self._coerce_injected_model(value)

    def initialize_model(self) -> None:
        """Construct the configured provider once, without making a remote request."""
        if self._model is not None:
            return
        kwargs = dict(self.kwargs)
        credential_data = dict(kwargs.pop("credential", {}) or {})
        if not _credential_api_key(credential_data):
            self.logger.warning(
                "%s:%s started without API credentials; LLM-backed jobs will fail until configured",
                self.component_type.value,
                self.name,
            )
            return
        credential = self.credential_cls(**credential_data)
        model_cls = credential.get_chat_model_class()
        params_dict = kwargs.pop("parameters", None)
        parameters = model_cls.Parameters(**params_dict) if params_dict else None
        self._model = model_cls(credential=credential, parameters=parameters, **kwargs)


@R.register("openai")
class OpenAIAsLLM(BaseAsLLM):
    """OpenAI chat model wrapper."""

    credential_cls = OpenAICredential


@R.register("anthropic")
class AnthropicAsLLM(BaseAsLLM):
    """Anthropic chat model wrapper."""

    credential_cls = AnthropicCredential


@R.register("dashscope")
class DashScopeAsLLM(BaseAsLLM):
    """DashScope chat model wrapper."""

    credential_cls = DashScopeCredential


@R.register("deepseek")
class DeepSeekAsLLM(BaseAsLLM):
    """DeepSeek chat model wrapper."""

    credential_cls = DeepSeekCredential


@R.register("gemini")
class GeminiAsLLM(BaseAsLLM):
    """Gemini chat model wrapper."""

    credential_cls = GeminiCredential


@R.register("moonshot")
class MoonshotAsLLM(BaseAsLLM):
    """Moonshot chat model wrapper."""

    credential_cls = MoonshotCredential


@R.register("ollama")
class OllamaAsLLM(BaseAsLLM):
    """Ollama chat model wrapper."""

    credential_cls = OllamaCredential


@R.register("xai")
class XAIAsLLM(BaseAsLLM):
    """xAI chat model wrapper."""

    credential_cls = XAICredential


@R.register("langchain")
class LangChainAsLLM(BaseAsLLM):
    """LangChain chat model wrapper.

    Construct from OpenAI-compatible credentials via ``langchain_openai.ChatOpenAI``,
    or inject a live LangChain / LlmProvider object through ``model=``.
    """

    credential_cls = OpenAICredential

    def initialize_model(self) -> None:
        """Build ``ChatOpenAI`` once, or keep an injected LangChain-compatible model."""
        if self._model is not None:
            return
        kwargs = dict(self.kwargs)
        credential_data = dict(kwargs.pop("credential", {}) or {})
        if not _credential_api_key(credential_data):
            self.logger.warning(
                "%s:%s started without API credentials; LLM-backed jobs will fail until configured",
                self.component_type.value,
                self.name,
            )
            return
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "as_llm backend 'langchain' requires langchain-openai; "
                "install with pip install 'reme-ai[langchain]' or langchain-openai",
            ) from exc

        model_name = kwargs.pop("model", None) or "gpt-4o-mini"
        params = dict(kwargs.pop("parameters", None) or {})
        client_kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": _credential_api_key(credential_data),
        }
        base_url = credential_data.get("base_url") or None
        if base_url:
            client_kwargs["base_url"] = base_url
        if params.get("temperature") is not None:
            client_kwargs["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            client_kwargs["max_tokens"] = params["max_tokens"]
        chat_model = ChatOpenAI(**client_kwargs)
        param_kwargs = {key: params[key] for key in ("temperature", "max_tokens") if key in params}
        self._model = LangChainChatModel(
            chat_model,
            model=model_name,
            stream=bool(kwargs.pop("stream", False)),
            context_size=int(kwargs.pop("context_size", 200000)),
            max_retries=int(kwargs.pop("max_retries", 3)),
            parameters=LangChainChatModel.Parameters(**param_kwargs),
        )


__all__ = [
    "BaseAsLLM",
    "OpenAIAsLLM",
    "AnthropicAsLLM",
    "DashScopeAsLLM",
    "DeepSeekAsLLM",
    "GeminiAsLLM",
    "MoonshotAsLLM",
    "OllamaAsLLM",
    "XAIAsLLM",
    "LangChainAsLLM",
    "LangChainChatModel",
    "coerce_to_chat_model",
    "is_langchain_compatible",
]
