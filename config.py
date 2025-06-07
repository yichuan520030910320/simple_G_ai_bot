# Configuration file for MapCrunch benchmark

SUCCESS_THRESHOLD_KM = 100

# MapCrunch settings
MAPCRUNCH_URL = "https://www.mapcrunch.com"

# UI element selectors
SELECTORS = {
    "go_button": "#go-button",
    "options_button": "#options-button",
    "stealth_checkbox": "#stealth",
    "urban_checkbox": "#cities",
    "indoor_checkbox": "#inside",
    "tour_checkbox": "#tour",
    "auto_checkbox": "#auto",
    "pano_container": "#pano",
    "map_container": "#map",
    "address_element": "#address",
    "confirm_button": "#confirm-button",  # Will be determined dynamically
    "country_list": "#countrylist",
    "continent_links": "#continents a",
}

# MapCrunch collection options
MAPCRUNCH_OPTIONS = {
    "urban_only": True,  # Show urban areas only
    "exclude_indoor": True,  # Exclude indoor views
    "stealth_mode": False,  # Hide location info during gameplay
    "tour_mode": False,  # 360 degree tour
    "auto_mode": False,  # Automatic slideshow
    "selected_countries": None,  # None means all, or list like ['us', 'gb', 'jp']
    "selected_continents": None,  # None means all, or list like [1, 2]  # 1=N.America, 2=Europe, etc
}

# Data collection settings
DATA_COLLECTION_CONFIG = {
    "save_thumbnails": True,  # Save small screenshots
    "thumbnail_size": (320, 240),  # Thumbnail dimensions
    "save_full_screenshots": False,  # Save full resolution screenshots (storage intensive)
    "extract_address": True,  # Extract address/location name
    "wait_after_go": 3,  # Seconds to wait after clicking Go
    "retry_on_failure": True,  # Retry if location fails
    "max_retries": 3,  # Max retries per location
}

# Reference points for coordinate calibration (used in pyautogui coordinate system)
REFERENCE_POINTS = {
    "kodiak": {"lat": 57.7916, "lon": -152.4083},
    "hobart": {"lat": -42.8833, "lon": 147.3355},
}

# Selenium settings
SELENIUM_CONFIG = {
    "headless": False,
    "window_size": (1920, 1080),
    "implicit_wait": 10,
    "page_load_timeout": 30,
}

# Model configurations
MODELS_CONFIG = {
    "gpt-4o": {
        "class": "ChatOpenAI",
        "model_name": "gpt-4o",
    },
    "claude-3.5-sonnet": {
        "class": "ChatAnthropic",
        "model_name": "claude-3-5-sonnet-20241022",
    },
    "gemini-1.5-pro": {
        "class": "ChatGoogleGenerativeAI",
        "model_name": "gemini-1.5-pro",
    },
}

# Benchmark settings
BENCHMARK_CONFIG = {
    "rounds_per_model": 50,
    "data_collection_samples": 200,
    "screenshot_delay": 2,
    "click_delay": 1,
}

# Data paths
DATA_PATHS = {
    "golden_labels": "data/golden_labels.json",
    "screenshots": "data/screenshots/",
    "thumbnails": "data/thumbnails/",
    "results": "results/",
    "screen_regions": "screen_regions.yaml",  # Keep for backward compatibility
}
