import os
import re

def rename_resized_frames_to_match_masks(images_dir):
    files = [f for f in os.listdir(images_dir) if f.startswith("resized_frame") and f.endswith(".jpg")]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    for f in files:
        idx = re.findall(r'\d+', f)[0]
        new_name = f"image_{idx}.jpg"
        old_path = os.path.join(images_dir, f)
        new_path = os.path.join(images_dir, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {f} -> {new_name}")

def rename_masks_to_match_resized_frames(masks_dir):
    files = [f for f in os.listdir(masks_dir) if f.startswith("image_") and f.endswith(".png")]
    files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    for f in files:
        idx = re.findall(r'\d+', f)[0]
        new_name = f"resized_frame{idx}.png"
        old_path = os.path.join(masks_dir, f)
        new_path = os.path.join(masks_dir, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {f} -> {new_name}")

if __name__ == "__main__":
    
    rename_resized_frames_to_match_masks("Data/Training/Fire")

    