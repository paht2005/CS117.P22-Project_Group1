import random
import datetime

def generate_gps():
    return {
        "lat": round(18.6765 + random.uniform(-0.001, 0.001), 6),
        "lon": round(105.7640 + random.uniform(-0.001, 0.001), 6),
        "alt": 75  # giả lập độ cao bay drone
    }

def estimate_fire_area(mask):
    sensor_width_mm = 6.3
    focal_length_mm = 24
    image_width_px = 1280
    R = (75 * sensor_width_mm) / (focal_length_mm * image_width_px)
    fire_pixels = (mask > 0).sum()
    return fire_pixels * (R ** 2)

def get_timestamp():
    return datetime.datetime.utcnow().isoformat() + "Z"
