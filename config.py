# Configuration file for MapCrunch benchmark

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
    "claude-3.5-sonnet": {
        "class": "ChatAnthropic",
        "model_name": "claude-3-5-sonnet-20240620",
        "description": "Anthropic Claude 3.5 Sonnet",
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
    "qwen2-vl-7b": {
        "class": "HuggingFaceChat",
        "model_name": "Qwen/Qwen2-VL-7B-Instruct",
        "description": "Qwen2-VL 7B (older but API supported)",
    },
    "qwen2-vl-2b": {
        "class": "HuggingFaceChat",
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "description": "Qwen2-VL 2B (faster, API supported)",
    },
}


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
