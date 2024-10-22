import pyautogui
from pynput import keyboard
import yaml


regions = [
    "screen_top_left",
    "screen_bot_right",
]

map_regions = [
    "map_top_left",
    "map_bot_right",
    "confirm_button",
    "kodiak",
    "hobart",
]

next_round_button = "next_round_button"

coords = []

PRESS_KEY = "a"


def on_press(key):
    try:
        if key.char == PRESS_KEY:
            x, y = pyautogui.position()
            print(x, y)
            coords.append([x, y])
            return False
    except AttributeError:
        pass


def get_coords(players=1):
    for region in regions:
        print(f"Move the mouse to the {region} region and press 'a'.")
        with keyboard.Listener(on_press=on_press) as keyboard_listener:
            keyboard_listener.join(timeout=40)
    
    for p in range(1, players+1):
        for region in map_regions:
            region = region + f"_{p}"
            regions.append(region)
            print(f"Move the mouse to the {region} region and press 'a'.")
            with keyboard.Listener(on_press=on_press) as keyboard_listener:
                keyboard_listener.join(timeout=40)

    regions.append(next_round_button)
    print(f"Move the mouse to the {next_round_button} region and press 'a'.")
    with keyboard.Listener(on_press=on_press) as keyboard_listener:
        keyboard_listener.join(timeout=40)

    screen_regions = {reg: coord for reg, coord in zip(regions, coords)}

    # save dict as a yaml file
    with open("screen_regions.yaml", "w") as f:
        yaml.dump(screen_regions, f)

    return screen_regions


if __name__ == "__main__":
    _ = get_coords(players=1)
