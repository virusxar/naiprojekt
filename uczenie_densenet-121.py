import torch
import torchvision.transforms as transforms
from torchvision import models, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

train_dir = "mushrooms-images-classification-215/data/data"
val_dir = "mushrooms-images-classification-215/data/data"

batch_size = 32  # Liczba próbek przetwarzanych jednocześnie
epochs = 60  # Liczba epok
learning_rate = 0.001  # Współczynnik uczenia
num_classes = 215  # Liczba klas w zbiorze danych(gatunki grzybów)

# Transformacje danych (zmiana rozmiaru, augmentacja, normalizacja)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Wczytanie danych treningowych i walidacyjnych
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

# Tworzenie loaderów danych, które umożliwiają iterację po danych w partiach (batchach)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

print("Ładowanie modelu DenseNet-121...")
model = models.densenet121(pretrained=True)  # Model wstępnie wytrenowany na ImageNet

# Modyfikacja klasyfikatora końcowego
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()  # Funkcja straty
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Optymalizator Adam

# Funkcja do trenowania modelu
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        running_loss = 0.0  # Suma strat dla epoki

        # Pętla po danych treningowych
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zerowanie gradientów
            outputs = model(inputs)  # Przepuszczenie danych przez model
            loss = criterion(outputs, labels)  # Obliczenie straty
            loss.backward()  # Obliczenie gradientów
            optimizer.step()  # Aktualizacja wag modelu

            running_loss += loss.item() * inputs.size(0)  # Sumowanie strat

        # Obliczenie średniej straty dla epoki
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Trening - strata: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)  # Przepuszczenie danych przez model
                loss = criterion(outputs, labels)  # Obliczenie straty
                val_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)  # Predykcja klasy z najwyższym prawdopodobieństwem
                correct += torch.sum(preds == labels)  # Liczenie poprawnych predykcji

        # Obliczenie średniej straty i dokładności walidacji
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct.double() / len(val_loader.dataset)
        print(f"Walidacja - strata: {val_loss:.4f}, dokładność: {val_acc:.4f}")

    print("Trening zakończony.")

if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs)

    # Zapisanie wytrenowanego modelu na dysku
    torch.save(model.state_dict(), "mushroom_classifier_densenet121.pth")
    print("Model zapisany jako mushroom_classifier_densenet121.pth")
