# Configuration file for MapCrunch benchmark

from pydantic import SecretStr, Field
from typing import Optional
import os


SUCCESS_THRESHOLD_KM = 100

# MapCrunch settings
MAPCRUNCH_URL = "https://www.mapcrunch.com"

# UI element selectors
SELECTORS = {
    "go_button": "#go-button",
    "pano_container": "#pano",
    "address_element": "#address",
}

# Data collection settings
DATA_COLLECTION_CONFIG = {
    "wait_after_go": 3,
    "thumbnail_size": (320, 240),
}

# Benchmark settings
BENCHMARK_CONFIG = {
    "data_collection_samples": 50,
}

# MapCrunch options
MAPCRUNCH_OPTIONS = {}

# Default settings
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_TEMPERATURE = 1.0

# Model configurations
MODELS_CONFIG = {
    "gpt-4o": {
        "class": "ChatOpenAI",
        "model_name": "gpt-4o",
        "description": "OpenAI GPT-4o",
    },
    "gpt-4o-mini": {
        "class": "ChatOpenAI",
        "model_name": "gpt-4o-mini",
        "description": "OpenAI GPT-4o Mini",
    },
    "claude-3-7-sonnet": {
        "class": "ChatAnthropic",
        "model_name": "claude-3-7-sonnet-20250219",
        "description": "Anthropic Claude 3.7 Sonnet",
    },
    "claude-4-sonnet": {
        "class": "ChatAnthropic",
        "model_name": "claude-4-sonnet-20250514",
        "description": "Anthropic Claude 4 Sonnet",
    },
    "gemini-1.5-pro": {
        "class": "ChatGoogleGenerativeAI",
        "model_name": "gemini-1.5-pro-latest",
        "description": "Google Gemini 1.5 Pro",
    },
    "gemini-2.0-flash-exp": {
        "class": "ChatGoogleGenerativeAI",
        "model_name": "gemini-2.0-flash-exp",
        "description": "Google Gemini 2.0 Flash Exp",
    },
    "gemini-2.5-pro": {
        "class": "ChatGoogleGenerativeAI",
        "model_name": "gemini-2.5-pro-preview-06-05",
        "description": "Google Gemini 2.5 Pro",
    },
    "qwen-vl-max": {
        "class": "OpenRouter",
        "model_name": "qwen/qwen-vl-max",
        "description": "Qwen VL Max - OpenRouter (Best Performance)",
    },
    "qwen2.5-vl-32b-free": {
        "class": "OpenRouter",
        "model_name": "qwen/qwen-2.5-vl-32b-instruct:free",
        "description": "Qwen2.5 VL 32B - OpenRouter (FREE!)",
    },
    "qwen2.5-vl-7b": {
        "class": "OpenRouter",
        "model_name": "qwen/qwen-2.5-vl-7b-instruct",
        "description": "Qwen2.5 VL 7B - OpenRouter",
    },
    "qwen2.5-vl-3b": {
        "class": "OpenRouter",
        "model_name": "qwen/qwen-2.5-vl-3b-instruct",
        "description": "Qwen2.5 VL 3B - OpenRouter (Fastest)",
    },
}

POSSIBLE_API_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "HF_TOKEN",
    "OPENROUTER_API_KEY",
]


def setup_environment_variables(st_secrets=None):
    for key in POSSIBLE_API_KEYS:
        # Try Streamlit secrets first if provided
        if st_secrets and key in st_secrets:
            os.environ[key] = st_secrets[key]
        elif key in os.environ:
            continue


def get_model_class(class_name):
    """Get actual model class from string name"""
    if class_name == "ChatOpenAI":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI
    elif class_name == "ChatAnthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic
    elif class_name == "ChatGoogleGenerativeAI":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI
    elif class_name == "HuggingFaceChat":
        from hf_chat import HuggingFaceChat

        return HuggingFaceChat
    elif class_name == "OpenRouter":
        from langchain_openai import ChatOpenAI
        from langchain_core.utils.utils import secret_from_env

        # LangChain does not support OpenRouter directly, so we need to create a custom class
        # See https://github.com/langchain-ai/langchain/discussions/27964.
        class ChatOpenRouter(ChatOpenAI):
            openai_api_key: Optional[SecretStr] = Field(
                alias="api_key",
                default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
            )

            @property
            def lc_secrets(self) -> dict[str, str]:
                return {"openai_api_key": "OPENROUTER_API_KEY"}

            def __init__(self, openai_api_key: Optional[str] = None, **kwargs):
                openai_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
                super().__init__(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=SecretStr(openai_api_key) if openai_api_key else None,
                    **kwargs,
                )

        return ChatOpenRouter
    else:
        raise ValueError(f"Unknown model class: {class_name}")


# Data paths - now supports named datasets
def get_data_paths(dataset_name: str = "default"):
    """Get data paths for a specific dataset"""
    return {
        "golden_labels": f"datasets/{dataset_name}/golden_labels.json",
        "thumbnails": f"datasets/{dataset_name}/thumbnails/",
        "results": f"results/{dataset_name}/",
    }


# Backward compatibility - default paths
DATA_PATHS = get_data_paths("default")
