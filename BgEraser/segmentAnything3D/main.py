# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import numpy as np
from PIL import Image

sys.path.append("notebook")
from infer import Inference, load_image, load_single_mask

# --- 暴力縮放函式 (只改長寬，不動通道內容) ---
def resize_simple(arr, size, is_mask=False):
    # 1. 轉 PIL (自動識別通道)
    pil_img = Image.fromarray(arr)
    
    # 2. 縮放 (遮罩用 NEAREST 防糊，圖片用 BILINEAR)
    algo = Image.NEAREST if is_mask else Image.BILINEAR
    pil_img = pil_img.resize(size, algo)
    
    # 3. 轉回 NumPy
    return np.array(pil_img)
# -----------------------------------------

tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)

# load image & mask
image = load_image("notebook/images/light_middle_barrel.png")

# 如果 image 是 RGBA
if image.ndim == 3 and image.shape[-1] == 4:
    alpha = image[:, :, 3]
    mask = (alpha > 0).astype(np.uint8)
    image = image[:, :, :3]  # 只留 RGB
else:
    raise ValueError("Image has no alpha channel, cannot auto-generate mask")

print(f"原始大小: {image.shape}, {mask.shape}")

# === 設定目標大小 (建議 512 或 1024) ===
target_size = (512, 512) 

# 1. 執行縮放
image = resize_simple(image, target_size, is_mask=False)
mask = resize_simple(mask, target_size, is_mask=True)

print(f"縮放後大小: {image.shape}, {mask.shape}")
# ======================================

# run model
output = inference(image, mask, seed=42)

# export gaussian splat
output["gs"].save_ply(f"splat.ply")
print("Your reconstruction has been saved to splat.ply")



