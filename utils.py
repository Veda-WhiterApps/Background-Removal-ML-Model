import cv2
import numpy as np

def refine_mask(mask):
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    final = (closed > 127).astype(np.uint8) * 255
    return final
