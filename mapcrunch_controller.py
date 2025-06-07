# mapcrunch_controller.py (Fixed)

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from typing import Dict, Optional
import time

# 修正: 从 config.py 导入所有需要的变量
from config import (
    MAPCRUNCH_URL,
    SELECTORS,
    DATA_COLLECTION_CONFIG,
    MAPCRUNCH_OPTIONS,
    SELENIUM_CONFIG,
)


class MapCrunchController:
    """Selenium controller for MapCrunch website automation"""

    def __init__(self, headless: bool = False):
        self.driver = None
        self.wait = None
        self.headless = headless
        self.setup_driver()

    def setup_driver(self):
        """Initialize Chrome driver with appropriate settings"""
        chrome_options = Options()

        if self.headless:
            chrome_options.add_argument("--headless")

        chrome_options.add_argument(
            f"--window-size={SELENIUM_CONFIG['window_size'][0]},{SELENIUM_CONFIG['window_size'][1]}"
        )
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_window_size(*SELENIUM_CONFIG["window_size"])
        self.wait = WebDriverWait(self.driver, SELENIUM_CONFIG["implicit_wait"])

        self.driver.get(MAPCRUNCH_URL)
        time.sleep(3)

    def setup_clean_environment(self):
        """
        Forcefully enables stealth mode and hides UI elements for a clean benchmark environment.
        """
        try:
            # 1. 强制开启 Stealth 模式
            # 这一步确保地址信息被网站自身的逻辑隐藏
            stealth_checkbox = self.wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["stealth_checkbox"])
                )
            )
            if not stealth_checkbox.is_selected():
                # 使用JS点击更可靠，可以避免元素被遮挡的问题
                self.driver.execute_script("arguments[0].click();", stealth_checkbox)
                print("✅ Stealth mode programmatically enabled for benchmark.")

            # 2. 用 JS 隐藏其他视觉干扰元素
            # 这一步确保截图区域干净
            self.driver.execute_script("""
                const elementsToHide = ['#menu', '#info-box', '#social', '#bottom-box', '#topbar'];
                elementsToHide.forEach(sel => {
                    const el = document.querySelector(sel);
                    if (el) el.style.display = 'none';
                });
                const panoBox = document.querySelector('#pano-box');
                if (panoBox) panoBox.style.height = '100vh';
            """)
            print("✅ Clean UI configured for benchmark.")

        except Exception as e:
            print(f"⚠️ Warning: Could not fully configure clean environment: {e}")

    # setup_collection_options 函数保持不变...
    def setup_collection_options(self, options: Dict = None):
        if options is None:
            options = MAPCRUNCH_OPTIONS
        try:
            options_button = self.wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, SELECTORS["options_button"])
                )
            )
            options_button.click()
            time.sleep(1)
            # ... (内部逻辑和之前一样)
            urban_checkbox = self.driver.find_element(
                By.CSS_SELECTOR, SELECTORS["urban_checkbox"]
            )
            if options.get("urban_only", False) != urban_checkbox.is_selected():
                urban_checkbox.click()
            indoor_checkbox = self.driver.find_element(
                By.CSS_SELECTOR, SELECTORS["indoor_checkbox"]
            )
            if options.get("exclude_indoor", True) == indoor_checkbox.is_selected():
                indoor_checkbox.click()
            stealth_checkbox = self.driver.find_element(
                By.CSS_SELECTOR, SELECTORS["stealth_checkbox"]
            )
            if options.get("stealth_mode", True) != stealth_checkbox.is_selected():
                stealth_checkbox.click()
            options_button.click()
            time.sleep(0.5)
            print("✅ Collection options configured")
            return True
        except Exception as e:
            print(f"❌ Error configuring options: {e}")
            return False

    def _select_countries(self, country_codes: list):
        """Select specific countries in the options panel"""
        try:
            # First, deselect all
            assert self.driver is not None
            all_countries = self.driver.find_elements(By.CSS_SELECTOR, "#countrylist a")
            for country in all_countries:
                class_attr = country.get_attribute("class")
                if class_attr is not None and "hover" not in class_attr:
                    country.click()
                    time.sleep(0.1)

            # Then select desired countries
            for code in country_codes:
                country = self.driver.find_element(
                    By.CSS_SELECTOR, f'a[data-code="{code}"]'
                )
                class_attr = country.get_attribute("class")
                if class_attr is not None and "hover" in class_attr:
                    country.click()
                    time.sleep(0.1)

            print(f"✅ Selected countries: {country_codes}")

        except Exception as e:
            print(f"⚠️  Warning: Could not select countries: {e}")

    def click_go_button(self) -> bool:
        """Click the Go button to get new Street View location"""
        try:
            go_button = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, SELECTORS["go_button"]))
            )
            go_button.click()
            # 修正: DATA_COLLECTION_CONFIG 现在已被导入，可以正常使用
            time.sleep(DATA_COLLECTION_CONFIG.get("wait_after_go", 5))
            return True
        except Exception as e:
            # 修正: 打印出具体的错误信息
            print(f"❌ Error clicking Go button: {e}")
            return False

    def get_current_address(self) -> Optional[str]:
        # ... (此函数不变) ...
        try:
            address_element = self.wait.until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["address_element"])
                )
            )
            address_text = address_element.text.strip()
            address_title = address_element.get_attribute("title") or ""
            return (
                address_title
                if len(address_title) > len(address_text)
                else address_text
            )
        except Exception:
            # 在stealth模式下，这个元素可能是隐藏的，所以找不到是正常的
            return "Stealth Mode"

    # **新增**: 重新加入 get_map_element_info 函数
    def get_map_element_info(self) -> Dict:
        """Get map element position and size for coordinate conversion."""
        try:
            assert self.wait is not None
            map_element = self.wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["map_container"])
                )
            )
            rect = map_element.rect
            location = map_element.location
            return {
                "x": location["x"],
                "y": location["y"],
                "width": rect["width"],
                "height": rect["height"],
                "element": map_element,
            }
        except Exception as e:
            # 这个函数在benchmark中不是必须的，只是GeoBot初始化需要，可以优雅地失败
            # print(f"⚠️ Could not get map element info: {e}")
            return {}

    def take_street_view_screenshot(self) -> Optional[bytes]:
        # ... (此函数不变) ...
        try:
            pano_element = self.wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["pano_container"])
                )
            )
            return pano_element.screenshot_as_png
        except Exception:
            return None

    def get_live_location_identifiers(self) -> Dict:
        # ... (此函数不变) ...
        try:
            assert self.driver is not None
            # 调用网站自己的JS函数来获取实时链接
            live_identifiers = self.driver.execute_script("""
                try {
                    return {
                        permLink: getPermLink(), // 调用网站自己的函数
                        panoId: window.panorama.getPano(),
                        urlString: urlSlug() // 调用网站自己的函数
                    };
                } catch (e) {
                    return { error: e.toString() };
                }
            """)
            return live_identifiers
        except Exception as e:
            print(f"❌ Error getting live identifiers: {e}")
            return {}

    def load_location_from_data(self, location_data: Dict) -> bool:
        # ... (此函数不变) ...
        try:
            url_to_load = location_data.get("perm_link") or location_data.get("url")
            if url_to_load and ("/p/" in url_to_load or "/s/" in url_to_load):
                print(f"✅ Loading location via perm_link: {url_to_load}")
                self.driver.get(url_to_load)
                time.sleep(4)
                return True

            # **备用方案: 根据坐标和视角手动构建链接 (来自您建议的格式)**
            lat = location_data.get("lat")
            lng = location_data.get("lng")
            if lat and lng:
                # 尝试从 identifiers 中获取视角信息
                pov = "232.46_-5_0"  # 默认视角
                # 注意: 采集时也应该保存 pov 信息，此处为简化
                url_slug = f"{lat}_{lng}_{pov}"
                url_to_load = f"{MAPCRUNCH_URL}/p/{url_slug}"
                print(f"✅ Loading location by constructing URL: {url_to_load}")
                self.driver.get(url_to_load)
                time.sleep(3)
                return True

            print(
                "⚠️  No valid location identifier (perm_link, url, or coords) found in data."
            )
            return False
        except Exception as e:
            print(f"❌ Error loading location: {e}")
            return False

    def close(self):
        # ... (此函数不变) ...
        if self.driver:
            self.driver.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
