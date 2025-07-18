import torch
import os
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from model import get_model
from dataset import SegmentationDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load dataset
human_ds = SegmentationDataset("data/human/images", "data/human/masks", transform)
nonhuman_ds = SegmentationDataset("data/nonhuman/images", "data/nonhuman/masks", transform)

full_dataset = ConcatDataset([human_ds, nonhuman_ds])
subset = Subset(full_dataset, range(5000))  # Small but diverse

loader = DataLoader(subset, batch_size=4, shuffle=True)

model = get_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
bce = torch.nn.BCELoss()

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

epochs = 15
os.makedirs("saved_models", exist_ok=True)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        out = model(img)['out']
        loss = bce(out, mask) + dice_loss(out, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"📅 Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "saved_models/deeplab_bg.pt")
print("Model saved: saved_models/deeplab_bg.pt")
