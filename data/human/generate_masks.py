# data/human/generate_masks.py
#Used to convert json files to images and save them into png in masks

import os, json
import numpy as np
import cv2

# Adjust folder paths if necessary
IMG_DIR = 'data/human/images'
JSON_DIR = 'data/human/json'
MASK_DIR = 'data/human/masks'
os.makedirs(MASK_DIR, exist_ok=True)

for fname in os.listdir(JSON_DIR):
    if not fname.endswith('.json'):
        continue
    path = os.path.join(JSON_DIR, fname)
    with open(path, 'r') as f:
        data = json.load(f)
    img_name = data.get('imagePath')
    img_path = os.path.join(IMG_DIR, img_name)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data.get('shapes', []):
        pts = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=255)

    # Save mask PNG, using same base filename
    base = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(MASK_DIR, base + '_mask.png')
    cv2.imwrite(out, mask)
    print("Saved mask:", out)
