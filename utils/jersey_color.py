import cv2
import numpy as np
from collections import Counter

def get_dominant_color(image):
    resized = cv2.resize(image, (50, 50))
    pixels = resized.reshape(-1, 3)
    pixels = [tuple(p) for p in pixels]
    most_common = Counter(pixels).most_common(1)[0][0]
    b, g, r = most_common

    if r > 150 and g < 100 and b < 100:
        return "red"
    elif b > 150 and g < 100 and r < 100:
        return "blue"
    elif g > 150 and r < 100 and b < 100:
        return "green"
    elif r > 200 and g > 200 and b < 100:
        return "yellow"
    else:
        return "unknown"
