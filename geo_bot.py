from io import BytesIO
import os
import dotenv
import base64
import pyautogui
import matplotlib.pyplot as plt
import math
from time import time, sleep
from typing import Tuple, List, Optional, Dict
from PIL import Image

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from mapcrunch_controller import MapCrunchController
from config import REFERENCE_POINTS

dotenv.load_dotenv()


PROMPT_INSTRUCTIONS = """
Try to predict where the image was taken.
First describe the relevant details in the image to do it.
List some regions and places where it could be.
Choose the most likely Country and City or Specific Location.
At the end, in the last line apart from the previous reasoning, write the Latitude and Longitude from that guessed location
using the following format, making sure that the coords are valid floats, without anything else and making sure to be consistent with the format:
Lat: XX.XXXX, Lon: XX.XXXX
"""


class GeoBot:
    """Enhanced GeoBot that works with MapCrunch via Selenium + PyAutoGUI hybrid approach"""
    
    prompt_instructions: str = PROMPT_INSTRUCTIONS

    def __init__(self, model=ChatOpenAI, model_name="gpt-4o", use_selenium=True):
        self.model = model(model=model_name)
        self.use_selenium = use_selenium
        self.controller = MapCrunchController() if use_selenium else None
        
        # Get screen and map regions
        if use_selenium:
            self._setup_screen_regions()
        else:
            # Fallback to manual regions (backward compatibility)
            self._load_manual_regions()
        
        # Reference points for coordinate calibration
        self.kodiak_lat, self.kodiak_lon = REFERENCE_POINTS['kodiak']['lat'], REFERENCE_POINTS['kodiak']['lon']
        self.hobart_lat, self.hobart_lon = REFERENCE_POINTS['hobart']['lat'], REFERENCE_POINTS['hobart']['lon']

    def _setup_screen_regions(self):
        """Setup screen regions using Selenium element positions"""
        try:
            # Get map element info
            map_info = self.controller.get_map_element_info()
            
            # Convert browser coordinates to screen coordinates
            self.map_x = map_info['x']
            self.map_y = map_info['y'] 
            self.map_w = map_info['width']
            self.map_h = map_info['height']
            
            # Set screen capture region (full window)
            window_size = self.controller.driver.get_window_size()
            self.screen_x, self.screen_y = 0, 0
            self.screen_w = window_size['width']
            self.screen_h = window_size['height']
            
            # Reference points for coordinate conversion (approximate map positions)
            # These would need to be calibrated for MapCrunch's specific map projection
            self.kodiak_x = self.map_x + int(self.map_w * 0.1)  # Approximate
            self.kodiak_y = self.map_y + int(self.map_h * 0.2)
            self.hobart_x = self.map_x + int(self.map_w * 0.9)
            self.hobart_y = self.map_y + int(self.map_h * 0.8)
            
            print(f"📍 Screen regions setup: Map({self.map_x},{self.map_y},{self.map_w},{self.map_h})")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not setup screen regions via Selenium: {e}")
            self._load_manual_regions()

    def _load_manual_regions(self):
        """Fallback to manual screen regions (backward compatibility)"""
        import yaml
        
        try:
            with open("screen_regions.yaml") as f:
                screen_regions = yaml.safe_load(f)
            
            self.screen_x, self.screen_y = screen_regions["screen_top_left"]
            self.screen_w = screen_regions["screen_bot_right"][0] - self.screen_x
            self.screen_h = screen_regions["screen_bot_right"][1] - self.screen_y
            
            self.map_x, self.map_y = screen_regions["map_top_left_1"]
            self.map_w = screen_regions["map_bot_right_1"][0] - self.map_x
            self.map_h = screen_regions["map_bot_right_1"][1] - self.map_y
            
            self.kodiak_x, self.kodiak_y = screen_regions["kodiak_1"]
            self.hobart_x, self.hobart_y = screen_regions["hobart_1"]
            
        except FileNotFoundError:
            print("❌ No screen_regions.yaml found and Selenium setup failed")
            raise

    @staticmethod
    def pil_to_base64(image: Image) -> str:
        """Convert PIL image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_base64_str

    @classmethod
    def create_message(cls, images_data: List[str]) -> HumanMessage:
        """Create message for LLM with images"""
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": cls.prompt_instructions,
                },
            ] + [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_data}"},
                }
            for img_data in images_data],
        )
        return message

    def extract_location_from_response(self, response: BaseMessage) -> Optional[Tuple[float, float]]:
        """Extract latitude and longitude from LLM response"""
        try:
            response_lines = response.content.split("\n")
            
            # Find the line with coordinates
            prediction_line = None
            for line in reversed(response_lines):
                if line.strip() and "lat" in line.lower() and "lon" in line.lower():
                    prediction_line = line.strip()
                    break
            
            if not prediction_line:
                print("❌ No coordinate line found in response")
                return None
            
            print(f"\n🎯 {self.model.__class__.__name__} Prediction: {prediction_line}")

            # Parse: "Lat: XX.XXXX, Lon: XX.XXXX"
            parts = prediction_line.split(",")
            if len(parts) != 2:
                print("❌ Invalid coordinate format")
                return None
            
            lat_str = parts[0].split(":")[1].strip()
            lon_str = parts[1].split(":")[1].strip()
            
            lat = float(lat_str)
            lon = float(lon_str)
            
            # Convert to screen coordinates
            x, y = self.lat_lon_to_screen_pixels(lat, lon)
            print(f"📍 Screen coordinates: ({x}, {y})")
            
            # Clamp to map bounds
            x = max(self.map_x, min(x, self.map_x + self.map_w))
            y = max(self.map_y, min(y, self.map_y + self.map_h))
            
            return x, y
        
        except Exception as e:
            print(f"❌ Error parsing response: {e}")
            return None

    @staticmethod
    def lat_to_mercator_y(lat: float) -> float:
        """Convert latitude to Mercator Y coordinate"""
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    def lat_lon_to_screen_pixels(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert latitude and longitude to screen pixel coordinates"""
        
        # Calculate the x pixel coordinate using longitude
        lon_diff_ref = (self.kodiak_lon - self.hobart_lon)
        lon_diff = (self.kodiak_lon - lon)
        x = abs(self.kodiak_x - self.hobart_x) * (lon_diff / lon_diff_ref) + self.kodiak_x

        # Calculate the y pixel coordinate using Mercator projection
        mercator_y1 = self.lat_to_mercator_y(self.kodiak_lat)
        mercator_y2 = self.lat_to_mercator_y(self.hobart_lat)
        mercator_y = self.lat_to_mercator_y(lat)

        lat_diff_ref = (mercator_y1 - mercator_y2)
        lat_diff = (mercator_y1 - mercator_y)
        y = abs(self.kodiak_y - self.hobart_y) * (lat_diff / lat_diff_ref) + self.kodiak_y

        return round(x), round(y)

    def take_screenshot(self) -> Image:
        """Take screenshot of the game area"""
        if self.use_selenium and self.controller:
            # Try Selenium screenshot first
            screenshot_bytes = self.controller.take_street_view_screenshot()
            if screenshot_bytes:
                return Image.open(BytesIO(screenshot_bytes))
        
        # Fallback to PyAutoGUI
        return pyautogui.screenshot(region=(self.screen_x, self.screen_y, self.screen_w, self.screen_h))

    def select_map_location(self, x: int, y: int, plot: bool = False) -> None:
        """Select location on map using hybrid Selenium + PyAutoGUI approach"""
        
        if self.use_selenium and self.controller:
            # Try Selenium click first (more reliable for web elements)
            screen_to_lat_lon = self._screen_pixels_to_lat_lon(x, y)
            if screen_to_lat_lon:
                lat, lon = screen_to_lat_lon
                if self.controller.click_map_location(lat, lon):
                    if plot:
                        self.plot_minimap(x, y)
                    return
        
        # Fallback to PyAutoGUI
        print("🖱️  Using PyAutoGUI for map click")
        
        # Hover over minimap to expand it
        pyautogui.moveTo(self.map_x + self.map_w - 15, self.map_y + self.map_h - 15, duration=0.5)
        sleep(1.5)

        # Click on predicted location
        pyautogui.click(x, y, duration=0.5)
        sleep(0.5)

        if plot:
            self.plot_minimap(x, y)

    def _screen_pixels_to_lat_lon(self, x: int, y: int) -> Optional[Tuple[float, float]]:
        """Convert screen pixels back to lat/lon (inverse of lat_lon_to_screen_pixels)"""
        try:
            # Reverse the x coordinate calculation
            x_ratio = (x - self.kodiak_x) / abs(self.kodiak_x - self.hobart_x)
            lon_diff = x_ratio * (self.kodiak_lon - self.hobart_lon)
            lon = self.kodiak_lon - lon_diff

            # Reverse the y coordinate calculation  
            y_ratio = (y - self.kodiak_y) / abs(self.kodiak_y - self.hobart_y)
            
            mercator_y1 = self.lat_to_mercator_y(self.kodiak_lat)
            mercator_y2 = self.lat_to_mercator_y(self.hobart_lat)
            lat_diff_ref = (mercator_y1 - mercator_y2)
            lat_diff = y_ratio * lat_diff_ref
            mercator_y = mercator_y1 - lat_diff
            
            # Convert back from Mercator to latitude
            lat = math.degrees(2 * math.atan(math.exp(mercator_y)) - math.pi/2)
            
            return lat, lon
            
        except Exception as e:
            print(f"❌ Error converting screen pixels to lat/lon: {e}")
            return None

    def plot_minimap(self, x: int = None, y: int = None) -> None:
        """Plot minimap with reference and prediction points"""
        minimap = pyautogui.screenshot(region=(self.map_x, self.map_y, self.map_w, self.map_h))
        
        plot_kodiak_x = self.kodiak_x - self.map_x
        plot_kodiak_y = self.kodiak_y - self.map_y
        plot_hobart_x = self.hobart_x - self.map_x
        plot_hobart_y = self.hobart_y - self.map_y
        
        plt.figure(figsize=(10, 8))
        plt.imshow(minimap)
        plt.plot(plot_hobart_x, plot_hobart_y, 'ro', markersize=8, label='Hobart (ref)')
        plt.plot(plot_kodiak_x, plot_kodiak_y, 'ro', markersize=8, label='Kodiak (ref)')
        
        if x and y:
            plt.plot(x - self.map_x, y - self.map_y, 'bo', markersize=10, label='Prediction')
        
        plt.legend()
        plt.title('Map with Predictions')
        
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/minimap.png", dpi=150, bbox_inches='tight')
        plt.close()

    def analyze_image(self, image: Image) -> Optional[Tuple[float, float]]:
        """Analyze image and return screen coordinates for predicted location"""
        try:
            screenshot_b64 = self.pil_to_base64(image)
            message = self.create_message([screenshot_b64])
            
            response = self.model.invoke([message])
            print(f"\n🤖 Full response from {self.model.__class__.__name__}:")
            print(response.content)
            
            location = self.extract_location_from_response(response)
            
            if location is None:
                print("🔄 Retrying analysis...")
                response = self.model.invoke([message])
                location = self.extract_location_from_response(response)
            
            return location
            
        except Exception as e:
            print(f"❌ Error in analyze_image: {e}")
            return None

    def close(self):
        """Clean up resources"""
        if self.controller:
            self.controller.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()