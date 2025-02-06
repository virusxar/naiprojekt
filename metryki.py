import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

val_dir = "mushrooms-images-classification-215/data/data"
batch_size = 32  # Liczba próbek przetwarzanych na raz w czasie jednej iteracji
num_classes = 215  # Liczba klas w problemie klasyfikacji

# Transformacja obrazów - przygotowanie obrazów wejściowych dla modelu
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Wczytywanie danych walidacyjnych
val_data = datasets.ImageFolder(val_dir, transform=test_transform)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Wczytanie wytrenowanego modelu
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Dostosowanie ostatniej warstwy do liczby klas
model.load_state_dict(torch.load("mushroom_classifier_resnet50.pth", map_location=device))  # Załadowanie wag modelu
model = model.to(device)
model.eval()

# Obliczanie metryk dla zbioru walidacyjnego
all_labels = []
all_preds = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)  # Wybranie klasy o najwyższym prawdopodobieństwie
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Obliczenie metryk Precision, Recall i F1-score
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")

# Wizualizacja metryk na wykresie słupkowym
metrics = [precision, recall, f1]
names = ['Precision', 'Recall', 'F1-score']
colors = ['blue', 'green', 'red']

plt.figure(figsize=(8, 6))
plt.bar(names, metrics, color=colors, width=0.4)
plt.ylim(0, 1)
plt.xlabel("Metryki", fontsize=14) 
plt.ylabel("Wartość", fontsize=14)
plt.title("resnet50\n Precision, Recall, F1-score", fontsize=16, pad=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

for i, v in enumerate(metrics):
    plt.text(i, v + 0.03, f"{v:.4f}", ha='center', fontsize=12, fontweight='bold')

plt.savefig("metrics_resnet50.png", dpi=300)
plt.show()

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
