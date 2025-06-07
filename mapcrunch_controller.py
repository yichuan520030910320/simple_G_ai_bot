# mapcrunch_controller.py (Fixed)

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from typing import Dict, Optional
import time

from config import MAPCRUNCH_URL, SELECTORS, DATA_COLLECTION_CONFIG, MAPCRUNCH_OPTIONS


class MapCrunchController:
    def __init__(self, headless: bool = False):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless")
        options.add_argument("--window-size=1920,1080")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)
        self.driver.get(MAPCRUNCH_URL)
        time.sleep(3)

    # **新增**: 完整实现了选项设置功能
    def setup_collection_options(self, options: Dict = None):
        if options is None:
            options = MAPCRUNCH_OPTIONS
        try:
            options_button = self.wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, SELECTORS["options_button"])
                )
            )
            # 点击以确保面板是打开的
            if "visible" not in options_button.find_element(
                By.XPATH, ".."
            ).get_attribute("class"):
                options_button.click()
            time.sleep(1)

            # Urban
            urban_checkbox = self.driver.find_element(
                By.CSS_SELECTOR, SELECTORS["urban_checkbox"]
            )
            if options.get("urban_only", False) != urban_checkbox.is_selected():
                urban_checkbox.click()
                print(f"✅ Urban mode set to: {options.get('urban_only', False)}")

            # Indoor
            indoor_checkbox = self.driver.find_element(
                By.CSS_SELECTOR, SELECTORS["indoor_checkbox"]
            )
            if options.get("exclude_indoor", True) == indoor_checkbox.is_selected():
                indoor_checkbox.click()
                print(
                    f"✅ Indoor views excluded: {options.get('exclude_indoor', True)}"
                )

            # 关闭面板
            options_button.click()
            time.sleep(0.5)
            print("✅ Collection options configured.")
            return True
        except Exception as e:
            print(f"❌ Error configuring options: {e}")
            return False

    # ... 其他所有函数 (click_go_button, get_live_location_identifiers, 等) 保持我们上一版的最终形态，无需改动 ...
    def click_go_button(self) -> bool:
        try:
            go_button = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, SELECTORS["go_button"]))
            )
            go_button.click()
            time.sleep(DATA_COLLECTION_CONFIG.get("wait_after_go", 5))
            return True
        except Exception as e:
            print(f"❌ Error clicking Go button: {e}")
            return False

    def get_live_location_identifiers(self) -> Dict:
        try:
            return self.driver.execute_script("""
                try {
                    const pov = window.panorama.getPov();
                    return {
                        panoId: window.panorama ? window.panorama.getPano() : null,
                        pov: { heading: pov.heading, pitch: pov.pitch, zoom: pov.zoom }
                    };
                } catch (e) { return { error: e.toString() }; }
            """)
        except Exception as e:
            print(f"❌ Error getting live identifiers via JS: {e}")
            return {}

    def get_current_address(self) -> Optional[str]:
        try:
            address_element = self.wait.until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["address_element"])
                )
            )
            return address_element.get_attribute("title") or address_element.text
        except TimeoutException:
            return "Address not found"

    def setup_clean_environment(self):
        try:
            self.driver.execute_script(
                "if(typeof hideLoc === 'function') { hideLoc(); }"
            )
            self.driver.execute_script("""
                const elementsToHide = ['#menu', '#social', '#bottom-box', '#topbar'];
                elementsToHide.forEach(sel => { const el = document.querySelector(sel); if (el) el.style.display = 'none'; });
                const panoBox = document.querySelector('#pano-box'); if (panoBox) panoBox.style.height = '100vh';
            """)
        except Exception as e:
            print(f"⚠️ Warning: Could not fully configure clean environment: {e}")

    def load_location_from_data(self, location_data: Dict) -> bool:
        try:
            url_slug = location_data.get("url_slug")
            if url_slug:
                url_to_load = f"{MAPCRUNCH_URL}/p/{url_slug}"
                print(f"✅ Loading location via precise URL Slug: {url_to_load}")
                self.driver.get(url_to_load)
                time.sleep(4)
                return True
            print("⚠️ No url_slug found in location data. Cannot load precisely.")
            return False
        except Exception as e:
            print(f"❌ Error loading location: {e}")
            return False

    def take_street_view_screenshot(self) -> Optional[bytes]:
        try:
            pano_element = self.wait.until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, SELECTORS["pano_container"])
                )
            )
            return pano_element.screenshot_as_png
        except Exception:
            return None

    def close(self):
        if self.driver:
            self.driver.quit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
