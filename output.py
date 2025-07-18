import torch, os, numpy as np, cv2
from PIL import Image
from torchvision import transforms
from model import UNet

model = UNet()
model.load_state_dict(torch.load("saved_models/unet_bg_removal.pt", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

os.makedirs("output", exist_ok=True)

for fname in os.listdir("test_images"):
    path = os.path.join("test_images", fname)
    img = Image.open(path).convert("RGB")
    orig = np.array(img)
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)[0, 0].numpy()

    mask = (pred > 0.5).astype(np.uint8)
    mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
    result = cv2.bitwise_and(orig, orig, mask=mask)
    out_path = f"output/{fname}"
    cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"✅ Output: {out_path}")
