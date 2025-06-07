# geo_bot.py (Final Streamlined Version)

from io import BytesIO
import base64
import re
from typing import Tuple, List, Optional
from PIL import Image

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from mapcrunch_controller import MapCrunchController

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
    """A streamlined bot focused purely on image analysis for the benchmark."""

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
        """Takes a screenshot of the Street View area using the controller."""
        if self.use_selenium and self.controller:
            screenshot_bytes = self.controller.take_street_view_screenshot()
            if screenshot_bytes:
                return Image.open(BytesIO(screenshot_bytes))
        return None

    def analyze_image(self, image: Image) -> Optional[Tuple[float, float]]:
        """Analyzes an image and returns the predicted (latitude, longitude)."""
        try:
            screenshot_b64 = self.pil_to_base64(image)
            message = self.create_message([screenshot_b64])

            response = self.model.invoke([message])
            print(f"\n🤖 Full response from {self.model_name}:")
            print(response.content)

            return self.extract_lat_lon_from_response(response)

        except Exception as e:
            print(f"❌ Error in analyze_image: {e}")
            return None

    def close(self):
        """Cleans up resources."""
        if self.controller:
            self.controller.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
