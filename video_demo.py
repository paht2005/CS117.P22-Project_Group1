
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Load the new video
video_path = "/demo/Drone_video.mp4"
cap = cv2.VideoCapture(video_path)

# Use only first 6 seconds for demo
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = 0
end_frame = int(6 * fps)

# Dummy metadata
fake_gps = {"lat": 18.6778, "lon": 105.7602, "alt": 120}
sensor_width_mm = 6.3
focal_length_mm = 24
image_width_px = 1280

input_frames, output_frames = [], []

while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    frame_resized = cv2.resize(frame, (512, 512))
    input_frames.append(frame_resized)

    # Fake fire region
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(mask, (256, 256), 70, 255, -1)

    fire_pixel_count = (mask > 0).sum()
    R = (fake_gps["alt"] * sensor_width_mm) / (focal_length_mm * image_width_px)
    area_m2 = fire_pixel_count * (R ** 2)

    overlay = frame_resized.copy()
    overlay[mask > 0] = [0.6 * overlay[mask > 0] + 0.4 * np.array([255, 0, 0])]

    pil_overlay = Image.fromarray(cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_overlay)
    draw.text((10, 10), "🔥 FIRE DETECTED", fill="red")
    draw.text((10, 30), f"Lat: {fake_gps['lat']} | Lon: {fake_gps['lon']}", fill="white")
    draw.text((10, 50), f"Alt: {fake_gps['alt']} m", fill="white")
    draw.text((10, 70), f"Estimated Area: {area_m2:.2f} m²", fill="white")
    overlay_final = cv2.cvtColor(np.array(pil_overlay), cv2.COLOR_RGB2BGR)
    output_frames.append(overlay_final)

cap.release()

# Export to video
input_path = "/demo/input_video.mp4"
output_path = "/demo/output_video.mp4"

out_input = cv2.VideoWriter(input_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (512, 512))
for frame in input_frames:
    out_input.write(frame)
out_input.release()

out_output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (512, 512))
for frame in output_frames:
    out_output.write(frame.astype(np.uint8))
out_output.release()

input_path, output_path
