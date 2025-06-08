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
}

# Model configurations
MODELS_CONFIG = {
    "gpt-4o": {
        "class": "ChatOpenAI",
        "model_name": "gpt-4o",
    },
    "claude-3.5-sonnet": {
        "class": "ChatAnthropic",
        "model_name": "claude-3-5-sonnet-20240620",
    },
    "gemini-1.5-pro": {
        "class": "ChatGoogleGenerativeAI",
        "model_name": "gemini-1.5-pro-latest",
    },
    "gemini-2.5-pro": {
        "class": "ChatGoogleGenerativeAI",
        "model_name": "gemini-2.5-pro-preview-06-05",
    },
}

# Data paths
DATA_PATHS = {
    "golden_labels": "data/golden_labels.json",
    "results": "results/",
}
