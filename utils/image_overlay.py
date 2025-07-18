from PIL import Image, ImageDraw
import cv2
import numpy as np

def overlay_mask(image_path, mask, gps, area, save_path):
    image = Image.open(image_path).convert("RGB").resize(mask.shape[::-1])
    image_np = np.array(image)
    image_np[mask > 0] = 0.6 * image_np[mask > 0] + 0.4 * np.array([255, 0, 0])

    result = Image.fromarray(image_np.astype(np.uint8))
    draw = ImageDraw.Draw(result)
    draw.text((10, 10), "🔥 FIRE DETECTED", fill="red")
    draw.text((10, 30), f"Lat: {gps['lat']} | Lon: {gps['lon']}", fill="white")
    draw.text((10, 50), f"Alt: {gps['alt']} m", fill="white")
    draw.text((10, 70), f"Estimated Area: {area:.2f} m²", fill="white")
    result.save(save_path)