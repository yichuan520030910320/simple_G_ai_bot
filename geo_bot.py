# geo_bot.py (Final Version)

from io import BytesIO
import os
import dotenv
import base64
import re  # 导入 re 模块
from typing import Tuple, List, Optional
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
    prompt_instructions: str = PROMPT_INSTRUCTIONS

    def __init__(
        self, model=ChatOpenAI, model_name="gpt-4o", use_selenium=True, headless=False
    ):
        self.model = model(model=model_name)
        self.model_name = model_name
        self.use_selenium = use_selenium
        self.controller = (
            MapCrunchController(headless=headless) if use_selenium else None
        )

        # Get screen and map regions
        if use_selenium:
            self._setup_screen_regions()
        else:
            # Fallback to manual regions (backward compatibility)
            self._load_manual_regions()

        # Reference points for coordinate calibration
        self.kodiak_lat, self.kodiak_lon = (
            REFERENCE_POINTS["kodiak"]["lat"],
            REFERENCE_POINTS["kodiak"]["lon"],
        )
        self.hobart_lat, self.hobart_lon = (
            REFERENCE_POINTS["hobart"]["lat"],
            REFERENCE_POINTS["hobart"]["lon"],
        )

    def _setup_screen_regions(self):
        """Setup screen regions using Selenium element positions"""
        try:
            # Get map element info
            map_info = self.controller.get_map_element_info()

            # Convert browser coordinates to screen coordinates
            self.map_x = map_info["x"]
            self.map_y = map_info["y"]
            self.map_w = map_info["width"]
            self.map_h = map_info["height"]

            # Set screen capture region (full window)
            window_size = self.controller.driver.get_window_size()
            self.screen_x, self.screen_y = 0, 0
            self.screen_w = window_size["width"]
            self.screen_h = window_size["height"]

            # Reference points for coordinate conversion (approximate map positions)
            # These would need to be calibrated for MapCrunch's specific map projection
            self.kodiak_x = self.map_x + int(self.map_w * 0.1)  # Approximate
            self.kodiak_y = self.map_y + int(self.map_h * 0.2)
            self.hobart_x = self.map_x + int(self.map_w * 0.9)
            self.hobart_y = self.map_y + int(self.map_h * 0.8)

            print(
                f"📍 Screen regions setup: Map({self.map_x},{self.map_y},{self.map_w},{self.map_h})"
            )

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
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @classmethod
    def create_message(cls, images_data: List[str]) -> HumanMessage:
        content = [{"type": "text", "text": cls.prompt_instructions}]
        for img_data in images_data:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_data}"},
                }
            )
        return HumanMessage(content=content)

    def extract_lat_lon_from_response(
        self, response: BaseMessage
    ) -> Optional[Tuple[float, float]]:
        """Extracts latitude and longitude from LLM response using regex for robustness."""
        try:
            content = response.content.strip()
            last_line = ""
            for line in reversed(content.split("\n")):
                if "lat" in line.lower() and "lon" in line.lower():
                    last_line = line
                    break

            if not last_line:
                print(f"❌ No coordinate line found in response.")
                return None

            print(f"🎯 {self.model_name} Prediction: {last_line}")

            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", last_line)

            if len(numbers) < 2:
                print(
                    f"❌ Could not find two numbers for lat/lon in line: '{last_line}'"
                )
                return None

            lat, lon = float(numbers[0]), float(numbers[1])

            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                print(f"❌ Invalid coordinates extracted: Lat {lat}, Lon {lon}")
                return None

            return lat, lon

        except Exception as e:
            print(
                f"❌ Error parsing lat/lon from response: {e}\nFull response was:\n{content}"
            )
            return None

    def take_screenshot(self) -> Optional[Image.Image]:
        if self.use_selenium and self.controller:
            screenshot_bytes = self.controller.take_street_view_screenshot()
            if screenshot_bytes:
                return Image.open(BytesIO(screenshot_bytes))
        return None

    def analyze_image(self, image: Image) -> Optional[Tuple[float, float]]:
        """Analyze image and return predicted latitude and longitude."""
        try:
            screenshot_b64 = self.pil_to_base64(image)
            message = self.create_message([screenshot_b64])

            response = self.model.invoke([message])
            print(f"\n🤖 Full response from {self.model_name}:")
            print(response.content)

            # 直接返回 (lat, lon) 元组
            return self.extract_lat_lon_from_response(response)

        except Exception as e:
            print(f"❌ Error in analyze_image: {e}")
            return None

    def close(self):
        if self.controller:
            self.controller.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
