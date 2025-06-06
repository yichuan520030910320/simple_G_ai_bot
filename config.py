# Configuration file for MapCrunch benchmark

# MapCrunch settings
MAPCRUNCH_URL = "https://www.mapcrunch.com"

# UI element selectors
SELECTORS = {
    'go_button': '#go-button',
    'options_button': '#options-button', 
    'stealth_checkbox': '#stealth',
    'pano_container': '#pano',
    'map_container': '#map',
    'address_element': '#address',
    'confirm_button': '#confirm-button',  # Will be determined dynamically
}

# Reference points for coordinate calibration (used in pyautogui coordinate system)
REFERENCE_POINTS = {
    'kodiak': {'lat': 57.7916, 'lon': -152.4083},
    'hobart': {'lat': -42.8833, 'lon': 147.3355}
}

# Selenium settings
SELENIUM_CONFIG = {
    'headless': False,
    'window_size': (1920, 1080),
    'implicit_wait': 10,
}

# Model configurations
MODELS_CONFIG = {
    'gpt-4o': {
        'class': 'ChatOpenAI',
        'model_name': 'gpt-4o',
    },
    'claude-3.5-sonnet': {
        'class': 'ChatAnthropic', 
        'model_name': 'claude-3-5-sonnet-20241022',
    },
    'gemini-1.5-pro': {
        'class': 'ChatGoogleGenerativeAI',
        'model_name': 'gemini-1.5-pro',
    }
}

# Benchmark settings
BENCHMARK_CONFIG = {
    'rounds_per_model': 50,
    'data_collection_samples': 200,
    'screenshot_delay': 2,
    'click_delay': 1,
}

# Data paths
DATA_PATHS = {
    'golden_labels': 'data/golden_labels.json',
    'screenshots': 'data/screenshots/',
    'results': 'results/',
    'screen_regions': 'screen_regions.yaml'  # Keep for backward compatibility
}