import torch, os, numpy as np, cv2
from PIL import Image
from torchvision import transforms
from model import get_model

model = get_model()
model.load_state_dict(torch.load("saved_models/deeplab_bg.pt", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

os.makedirs("output", exist_ok=True)

while True:
    path = input("📸 Enter image path (or 'exit'): ").strip()
    if path.lower() == 'exit': break
    if not os.path.exists(path):
        print("❌ File not found.")
        continue

    img = Image.open(path).convert("RGB")
    orig_np = np.array(img)
    orig_w, orig_h = img.size

    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(input_tensor)['out'][0, 0].numpy()

    mask = (out > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    rgba = cv2.cvtColor(orig_np, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask_resized

    name = os.path.splitext(os.path.basename(path))[0]
    output_path = f"output/{name}_cleaned.png"
    cv2.imwrite(output_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    print(f"✅ Saved: {output_path}")
