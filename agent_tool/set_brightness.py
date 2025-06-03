import screen_brightness_control as sbc
import argparse
def set_brightness(level):
    """
    Adjust the screen brightness to the specified level.
    :param level: Brightness level (0 to 100)
    """
    try:
        sbc.set_brightness(level)
        #print(f"Brightness set to {level}%")
        return f"✅ The brightness is {level}."
    except Exception as e:
        print(f"Failed to set brightness: {e}")