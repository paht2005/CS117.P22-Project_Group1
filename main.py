from fire_classifier import classify_fire
from fire_segmenter import segment_fire
from utils.image_overlay import overlay_mask
from utils.metadata import generate_gps, estimate_fire_area, get_timestamp
import os

input_dir = "data/Test"
output_dir = "data/Outputs"
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.lower().endswith((".jpg", ".png")):
        continue
    img_path = os.path.join(input_dir, fname)
    print(f"🔍 Đang xử lý: {fname}")

    if classify_fire(img_path):
        mask = segment_fire(img_path)
        gps = generate_gps()
        area = estimate_fire_area(mask)
        overlay_mask(img_path, mask, gps, area, os.path.join(output_dir, fname))
        print(f"🔥 Có cháy. Đã lưu kết quả.\n")
    else:
        print("✅ Không phát hiện cháy.\n")
