# mapcrunch_controller.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import time
from typing import Dict, Optional, Tuple
from config import MAPCRUNCH_URL, SELECTORS, SELENIUM_CONFIG


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
        """Configure MapCrunch for clean benchmark environment"""
        try:
            assert self.driver is not None
            self.driver.execute_script("""
                const elementsToHide = ['#menu', '#info-box', '#social', '#bottom-box'];
                elementsToHide.forEach(sel => {
                    const el = document.querySelector(sel);
                    if (el) el.style.display = 'none';
                });
            """)
            print("✅ Environment configured for clean benchmark")
        except Exception as e:
            print(f"⚠️  Warning: Could not fully configure environment: {e}")

    def setup_collection_options(self, options: Dict = None):
        from config import MAPCRUNCH_OPTIONS

        if options is None:
            options = MAPCRUNCH_OPTIONS
        try:
            assert self.wait is not None
            options_button = self.wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, SELECTORS["options_button"])
                )
            )
            options_button.click()
            time.sleep(1)

            assert self.driver is not None
            # Urban
            urban_checkbox = self.driver.find_element(
                By.CSS_SELECTOR, SELECTORS["urban_checkbox"]
            )
            if options.get("urban_only", False) != urban_checkbox.is_selected():
                urban_checkbox.click()

            # Indoor
            indoor_checkbox = self.driver.find_element(
                By.CSS_SELECTOR, SELECTORS["indoor_checkbox"]
            )
            if options.get("exclude_indoor", True) == indoor_checkbox.is_selected():
                indoor_checkbox.click()

            # Stealth
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
            assert self.wait is not None
            go_button = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, SELECTORS["go_button"]))
            )
            go_button.click()
            # **重要**: 等待JS执行完毕并更新内容
            time.sleep(DATA_COLLECTION_CONFIG.get("wait_after_go", 5))
            return True
        except Exception as e:
            print(f"❌ Error clicking Go button: {e}")
            return False

    def get_current_address(self) -> Optional[str]:
        """Extract current address/location name from the page"""
        try:
            assert self.wait is not None
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
        """Take screenshot of the Street View area"""
        try:
            assert self.wait is not None
            pano_element = self.wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["pano_container"])
                )
            )
            return pano_element.screenshot_as_png
        except Exception as e:
            print(f"❌ Error taking screenshot: {e}")
            return None

    # **新增**: 获取实时页面标识符的方法
    def get_live_location_identifiers(self) -> Dict:
        """Executes JS to get the identifiers of the CURRENTLY displayed location."""
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

    # **修改**: 增强 load_location_from_data
    def load_location_from_data(self, location_data: Dict) -> bool:
        """Load a specific location by navigating to its permanent link."""
        try:
            assert self.driver is not None

            # **优先使用 perm_link 或 url (现在应该已经是正确的了)**
            url_to_load = location_data.get("perm_link") or location_data.get("url")

            if url_to_load and "/p/" in url_to_load:
                print(f"✅ Loading location via perm_link: {url_to_load}")
                self.driver.get(url_to_load)
                time.sleep(3)  # 等待场景加载
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
        if self.driver:
            self.driver.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
