"""Central LLM factory.

All agents share one configuration so switching model/provider is a one-file
change. We talk to DeepSeek through OpenRouter, which exposes an OpenAI-compatible
API, so we use ChatOpenAI pointed at the OpenRouter base URL.
"""

import os

from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-pro"


def build_llm(schema):
    """Return a chat model bound to the given Pydantic schema via structured output.

    Configured entirely from the environment:
      - OPENROUTER_API_KEY   (required) — OpenRouter credential
      - LLM_MODEL            (optional) — model id, default deepseek/deepseek-v4-pro
      - OPENROUTER_BASE_URL  (optional) — override the API base
      - LLM_STRUCTURED_METHOD(optional) — with_structured_output method
                                          (function_calling | json_schema | json_mode)
    """
    llm = ChatOpenAI(
        model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
    )
    method = os.environ.get("LLM_STRUCTURED_METHOD", "function_calling")
    return llm.with_structured_output(schema, method=method)
