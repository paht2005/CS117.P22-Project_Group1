import os
from PIL import Image

# Đường dẫn thư mục
input_dir = 'data/Masks'         
output_dir = 'data/Resized_Masks'       
target_size = (254, 254)

# Tạo thư mục output nếu chưa có
os.makedirs(output_dir, exist_ok=True)

# Resize và lưu ảnh
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        # Load ảnh, resize và lưu
        img = Image.open(in_path).convert("L").resize(target_size)
        img.save(out_path)

        print(f" Resized and saved: {filename}")

print(" All masks resized")
