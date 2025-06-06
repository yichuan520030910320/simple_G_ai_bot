from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import time
import json
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
        
        chrome_options.add_argument(f"--window-size={SELENIUM_CONFIG['window_size'][0]},{SELENIUM_CONFIG['window_size'][1]}")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.set_window_size(*SELENIUM_CONFIG['window_size'])
        self.wait = WebDriverWait(self.driver, SELENIUM_CONFIG['implicit_wait'])
        
        # Navigate to MapCrunch
        self.driver.get(MAPCRUNCH_URL)
        time.sleep(3)  # Allow page to load
    
    def setup_clean_environment(self):
        """Configure MapCrunch for clean benchmark environment"""
        try:
            # Enable stealth mode (hide location info)
            stealth_checkbox = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, SELECTORS['stealth_checkbox']))
            )
            if not stealth_checkbox.is_selected():
                stealth_checkbox.click()
            
            # Hide additional UI elements via JavaScript
            self.driver.execute_script("""
                // Hide menu and info elements
                const menu = document.querySelector('#menu');
                if (menu) menu.style.display = 'none';
                
                const infoBox = document.querySelector('#info-box');
                if (infoBox) infoBox.style.display = 'none';
                
                const socialBox = document.querySelector('#social');
                if (socialBox) socialBox.style.display = 'none';
                
                // Make street view arrows larger (if available)
                const svLinks = document.querySelectorAll('.gm-style-moc');
                svLinks.forEach(link => {
                    link.style.fontSize = '24px';
                    link.style.fontWeight = 'bold';
                });
            """)
            
            print("✅ Environment configured for clean benchmark")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not fully configure environment: {e}")
    
    def click_go_button(self) -> bool:
        """Click the Go button to get new Street View location"""
        try:
            go_button = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, SELECTORS['go_button']))
            )
            go_button.click()
            time.sleep(3)  # Wait for new location to load
            return True
        except Exception as e:
            print(f"❌ Error clicking Go button: {e}")
            return False
    
    def get_current_coordinates(self) -> Optional[Dict[str, float]]:
        """Extract current Street View coordinates from MapCrunch page variables"""
        try:
            # Extract coordinates from MapCrunch JavaScript variables
            coords = self.driver.execute_script("""
                try {
                    // Method 1: Parse initString variable (primary method for MapCrunch)
                    if (typeof initString !== 'undefined' && initString) {
                        // initString format: "lat_lon_heading_pitch_zoom"
                        const parts = initString.split('_');
                        if (parts.length >= 2) {
                            const lat = parseFloat(parts[0]);
                            const lng = parseFloat(parts[1]);
                            if (!isNaN(lat) && !isNaN(lng)) {
                                return {
                                    lat: lat,
                                    lng: lng,
                                    source: 'mapcrunch_initString',
                                    address: typeof initLocationString !== 'undefined' ? initLocationString : null
                                };
                            }
                        }
                    }
                    
                    // Method 2: Try to get from Street View panorama object
                    if (window.google && window.google.maps && window.panorama) {
                        const position = window.panorama.getPosition();
                        if (position) {
                            return {
                                lat: position.lat(),
                                lng: position.lng(),
                                source: 'streetview_api',
                                address: null
                            };
                        }
                    }
                    
                    // Method 3: Try to parse from page content
                    const addressElement = document.querySelector('#address');
                    if (addressElement) {
                        const addressText = addressElement.textContent || addressElement.getAttribute('title');
                        if (addressText) {
                            return {
                                lat: null,
                                lng: null,
                                source: 'address_only',
                                address: addressText
                            };
                        }
                    }
                    
                    return null;
                } catch (error) {
                    console.error('Error getting coordinates:', error);
                    return null;
                }
            """)
            
            if coords and coords.get('lat') is not None and coords.get('lng') is not None:
                return coords
            
            # If we only got address, that's still useful
            if coords and coords.get('address'):
                return coords
            
            return None
            
        except Exception as e:
            print(f"❌ Error getting coordinates: {e}")
            return None
    
    def get_map_element_info(self) -> Dict:
        """Get map element position and size for coordinate conversion"""
        try:
            map_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, SELECTORS['map_container']))
            )
            
            # Get element rect info
            rect = map_element.rect
            location = map_element.location
            
            return {
                'x': location['x'],
                'y': location['y'], 
                'width': rect['width'],
                'height': rect['height'],
                'element': map_element
            }
        except Exception as e:
            print(f"❌ Error getting map element info: {e}")
            return {}
    
    def take_street_view_screenshot(self) -> Optional[bytes]:
        """Take screenshot of the Street View area"""
        try:
            pano_element = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, SELECTORS['pano_container']))
            )
            return pano_element.screenshot_as_png
        except Exception as e:
            print(f"❌ Error taking screenshot: {e}")
            return None
    
    def click_map_location(self, lat: float, lon: float) -> bool:
        """Click on map at specified coordinates"""
        try:
            map_info = self.get_map_element_info()
            if not map_info:
                return False
            
            # Convert lat/lon to pixel offset within map element
            x_offset, y_offset = self._lat_lon_to_map_offset(
                lat, lon, map_info['width'], map_info['height']
            )
            
            # Click using ActionChains
            ActionChains(self.driver).move_to_element_with_offset(
                map_info['element'], x_offset, y_offset
            ).click().perform()
            
            time.sleep(1)
            return True
            
        except Exception as e:
            print(f"❌ Error clicking map: {e}")
            return False
    
    def _lat_lon_to_map_offset(self, lat: float, lon: float, map_width: int, map_height: int) -> Tuple[int, int]:
        """Convert lat/lon to pixel offset within map element (Web Mercator projection)"""
        import math
        
        # Convert longitude (-180 to 180) to x offset (0 to map_width)
        x_offset = (lon + 180) * (map_width / 360)
        
        # Convert latitude to Web Mercator y offset
        lat_rad = math.radians(lat)
        mercator_y = math.log(math.tan(math.pi/4 + lat_rad/2))
        
        # Normalize to map height (assuming typical web map bounds)
        max_mercator = math.log(math.tan(math.pi/4 + math.radians(85)/2))  # ~85 degrees max
        y_offset = map_height * (1 - (mercator_y + max_mercator) / (2 * max_mercator))
        
        return int(x_offset), int(y_offset)
    
    def load_location_from_data(self, location_data: Dict) -> bool:
        """Load a specific location from collected data"""
        try:
            # Method 1: Use permanent link if available
            perm_link = location_data.get('perm_link')
            if perm_link:
                self.driver.get(perm_link)
                time.sleep(2)
                return True
            
            # Method 2: Reconstruct URL with initString if available
            init_string = location_data.get('init_string')
            if init_string:
                # Navigate to MapCrunch with specific parameters
                url = f"{MAPCRUNCH_URL}?{init_string}"
                self.driver.get(url)
                time.sleep(2)
                return True
                
            # Method 3: Use saved URL
            saved_url = location_data.get('url')
            if saved_url:
                self.driver.get(saved_url)
                time.sleep(2)
                return True
                
            # Method 4: Navigate using coordinates (if MapCrunch supports it)
            lat = location_data.get('lat')
            lng = location_data.get('lng')
            if lat and lng:
                # Try to navigate to coordinates (this may not work for all sites)
                url = f"{MAPCRUNCH_URL}?lat={lat}&lng={lng}"
                self.driver.get(url)
                time.sleep(2)
                return True
            
            print("⚠️  No valid location identifier found in data")
            return False
            
        except Exception as e:
            print(f"❌ Error loading location: {e}")
            return False
    
    def close(self):
        """Clean up and close browser"""
        if self.driver:
            self.driver.quit()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()