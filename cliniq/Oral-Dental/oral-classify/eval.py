import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_resnet50_oral.pth"
DATA_DIR = "dataset"
NUM_CLASSES = 6

# Transforms (same as training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Dataset
test_ds = datasets.ImageFolder(f"{DATA_DIR}/test", transform=test_tf)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

# Model
weights = models.ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Evaluation
preds = []
actual = []

with torch.no_grad():
    for imgs, labels in test_dl:
        imgs = imgs.to(DEVICE)
        out = model(imgs)
        pred = torch.argmax(out, dim=1).cpu().numpy()
        preds.extend(pred)
        actual.extend(labels.numpy())

print("\nCLASSIFICATION REPORT:")
print(classification_report(actual, preds, target_names=test_ds.classes))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(actual, preds))
