import pyautogui
import yaml
import os
from time import sleep

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from select_regions import get_coords
from geoguessr_bot import GeoBot


def play_turn(bot: GeoBot, plot: bool = False):
    screenshot = pyautogui.screenshot(region=bot.screen_xywh)
    screenshot_b64 = GeoBot.pil_to_base64(screenshot)
    message = GeoBot.create_message([screenshot_b64])

    response = bot.model.invoke([message])
    print(response.content)

    location = bot.extract_location_from_response(response)
    if location is None:
        # Second try
        response = bot.model.invoke([message])
        print(response.content)
        location = bot.extract_location_from_response(response)
    
    if location is not None:
        bot.select_map_location(*location, plot=plot)
    else:
        print("Error getting a location for second time")
        # TODO: strange default location, may neeed to change
        bot.select_map_location(x=1, y=1, plot=plot)

    # Going to the next round
    pyautogui.press(" ")
    sleep(2)


def main(turns=5, plot=False):
    if "screen_regions.yaml" not in os.listdir():
        screen_regions = get_coords(players=1)
    with open("screen_regions.yaml") as f:
        screen_regions = yaml.safe_load(f)

    bot = GeoBot(
        screen_regions=screen_regions,
        player=1,
        model=ChatOpenAI,  # ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropic
        model_name="gpt-4o",   # gpt-4o, gemini-1.5-pro, claude-3-5-sonnet-20240620
    )

    for turn in range(turns):
        print("\n----------------")
        print(f"Turn {turn+1}/{turns}")
        play_turn(bot=bot, plot=plot)


if __name__ == "__main__":
    main(turns=5, plot=True)