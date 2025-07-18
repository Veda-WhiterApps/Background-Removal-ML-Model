import cv2
m = cv2.imread("data/human/masks/2_mask.png", 0)
print("Unique pixel values in mask:", set(m.flatten()))