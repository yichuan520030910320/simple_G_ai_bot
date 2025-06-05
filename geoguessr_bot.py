from io import BytesIO
import os
import dotenv
import base64
import pyautogui
import matplotlib.pyplot as plt
import math
from time import time, sleep
from typing import Tuple, List  
from PIL import Image

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

dotenv.load_dotenv()


PROMPT_INSTRUCTIONS = """
Try to predict where the image was taken.
First describe the relevant details in the image to do it.
List some regions and places where it could be.
Chose the most likely Country and City or Specific Location.
At the end, in the last line a part from the previous reasoning, write the Latitude and Longitude from that guessed location
using the following format, making sure that the coords are valid floats, without anything else and making sure to be consistent with the format:
Lat: XX.XXXX, Lon: XX.XXXX
"""


class GeoBot:
    prompt_instructions: str = PROMPT_INSTRUCTIONS

    def __init__(self, screen_regions, player=1, model=ChatOpenAI, model_name="gpt-4o"):
        self.player = player
        self.screen_regions = screen_regions
        self.screen_x, self.screen_y = screen_regions["screen_top_left"]
        self.screen_w = screen_regions["screen_bot_right"][0] - self.screen_x
        self.screen_h = screen_regions["screen_bot_right"][1] - self.screen_y
        self.screen_xywh = (self.screen_x, self.screen_y, self.screen_w, self.screen_h)

        self.map_x, self.map_y = screen_regions[f"map_top_left_{player}"]
        self.map_w = screen_regions[f"map_bot_right_{player}"][0] - self.map_x
        self.map_h = screen_regions[f"map_bot_right_{player}"][1] - self.map_y
        self.minimap_xywh = (self.map_x, self.map_y, self.map_w, self.map_h)

        self.next_round_button = screen_regions["next_round_button"] if player==1 else None
        self.confirm_button = screen_regions[f"confirm_button_{player}"]

        self.kodiak_x, self.kodiak_y = screen_regions[f"kodiak_{player}"] 
        self.hobart_x, self.hobart_y = screen_regions[f"hobart_{player}"] 
        
        # Refernece points to calibrate the minimap everytime
        self.kodiak_lat, self.kodiak_lon = (57.7916, -152.4083)
        self.hobart_lat, self.hobart_lon = (-42.8833, 147.3355)

        self.model = model(model=model_name)


    @staticmethod
    def pil_to_base64(image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img_base64_str
    

    @classmethod
    def create_message(cls, images_data: List[str]) -> HumanMessage:
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
    

    def extract_location_from_response(self, response: BaseMessage) -> Tuple[float, float]:
        try:
            response = response.content.split("\n")
            while response and len(response[-1]) == 0 and "lat" not in response[-1].lower():
                response.pop()
            if response:
                prediction = response[-1]
            else:
                return None
            print(f"\n-------\n{self.model} Prediction:\n", prediction)

            # Lat: 57.7916, Lon: -152.4083
            lat = float(prediction.split(",")[0].split(":")[1])
            lon = float(prediction.split(",")[1].split(":")[1])

            x, y = self.lat_lon_to_mercator_map_pixels(lat, lon)
            print(f"Normalized pixel coordinates: ({x}, {y})")

            if x < self.map_x:
                x = self.map_x
                print("x out of bounds")
            elif x > self.map_x+self.map_w:
                x = self.map_x+self.map_w
                print("x out of bounds")
            if y < self.map_y:
                y = self.map_y
                print("y out of bounds")
            elif y > self.map_y+self.map_h:
                y = self.map_y+self.map_h
                print("y out of bounds")

            return x, y
        
        except Exception as e:
            print("Error:", e)
            return None
        

    @staticmethod
    def lat_to_mercator_y(lat: float) -> float:
        return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    

    def lat_lon_to_mercator_map_pixels(self, lat: float, lon: float) -> Tuple[int, int]:
        """ 
        Convert latitude and longitude to pixel coordinates on the mercator projection minimap, 
        taking two known points 1 and 2 as a reference.

        Args:
            lat (float): Latitude (Decimal Degrees) of the point to convert.
            lon (float): Longitude (Decimal Degrees) of the point to convert.

        Returns:
            tuple: x, y pixel coordinates of the point.
        """

        # Calculate the x pixel coordinate
        lon_diff_ref = (self.kodiak_lon - self.hobart_lon)
        lon_diff = (self.kodiak_lon - lon)

        x = abs(self.kodiak_x - self.hobart_x) * (lon_diff / lon_diff_ref) + self.kodiak_x

        # Convert latitude and longitude to mercator projection y coordinates
        mercator_y1 = self.lat_to_mercator_y(self.kodiak_lat)
        mercator_y2 = self.lat_to_mercator_y(self.hobart_lat)
        mercator_y = self.lat_to_mercator_y(lat)

        # Calculate the y pixel coordinate
        lat_diff_ref = (mercator_y1 - mercator_y2)
        lat_diff = (mercator_y1 - mercator_y)

        y = abs(self.kodiak_y - self.hobart_y) * (lat_diff / lat_diff_ref) + self.kodiak_y

        return round(x), round(y)
    

    def select_map_location(self, x: int, y: int, plot: bool = False) -> None:
        # Hovering over the minimap to expand it
        pyautogui.moveTo(self.map_x+self.map_w-15, self.map_y+self.map_h-15, duration=0.5)
        #bot.screen_w-50, bot.screen_h-80
        # pyautogui.moveTo(self.screen_w-50, self.screen_h-80, duration=1.5)
        # print(self.screen_w-50, self.screen_h-80)
        print('finish moving')
        sleep(0.5)

        # Clicking on the predicted location
        pyautogui.click(x, y, duration=0.5)
        print('finish clicking')
        sleep(0.5)

        if plot:
            self.plot_minimap(x, y)

        # Confirming the guessed location
        pyautogui.click(self.confirm_button, duration=0.2)
        sleep(2)

    
    def plot_minimap(self, x: int = None, y: int = None) -> None:
        minimap = pyautogui.screenshot(region=self.minimap_xywh)  
        plot_kodiak_x = self.kodiak_x - self.map_x
        plot_kodiak_y = self.kodiak_y - self.map_y
        plot_hobart_x = self.hobart_x - self.map_x
        plot_hobart_y = self.hobart_y - self.map_y
        plt.imshow(minimap)
        plt.plot(plot_hobart_x, plot_hobart_y, 'ro')
        plt.plot(plot_kodiak_x, plot_kodiak_y, 'ro')
        if x and y:
            plt.plot(x-self.map_x, y-self.map_y, 'bo')

        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/minimap.png")
        # plt.show()