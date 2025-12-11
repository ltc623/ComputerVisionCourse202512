"""
貓狗分類器 - 訓練腳本
使用 PyTorch 建立 CNN 模型進行貓狗二分類
支援使用預訓練模型 (Transfer Learning)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

# ============== 取得腳本所在目錄 (相對路徑基準) ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============== 設定區 ==============
BATCH_SIZE = 32          # 批次大小
EPOCHS = 10              # 訓練輪數
LEARNING_RATE = 0.001    # 學習率
IMAGE_SIZE = 224         # 影像大小 (配合預訓練模型)
TRAIN_SPLIT = 0.8        # 訓練集比例
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 相對路徑設定
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "catdog_model.pth")
USE_PRETRAINED = True                  # 是否使用預訓練模型


# ============== 資料準備 ==============
def setup_data_directory():
    """
    設置資料目錄結構
    需要的結構:
    data/
    ├── cats/
    │   ├── cat1.jpg
    │   ├── cat2.jpg
    │   └── ...
    └── dogs/
        ├── dog1.jpg
        ├── dog2.jpg
        └── ...
    """
    cats_dir = os.path.join(DATA_DIR, "cats")
    dogs_dir = os.path.join(DATA_DIR, "dogs")

    os.makedirs(cats_dir, exist_ok=True)
    os.makedirs(dogs_dir, exist_ok=True)

    # 檢查是否有資料
    cat_count = len([f for f in os.listdir(cats_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    dog_count = len([f for f in os.listdir(dogs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f"資料目錄: {DATA_DIR}")
    print(f"  貓圖片數量: {cat_count}")
    print(f"  狗圖片數量: {dog_count}")

    if cat_count == 0 or dog_count == 0:
        print("\n" + "=" * 50)
        print("警告: 資料目錄中沒有足夠的圖片!")
        print("請將貓的圖片放入 data/cats/ 目錄")
        print("請將狗的圖片放入 data/dogs/ 目錄")
        print("=" * 50)
        print("\n您可以從以下來源下載資料集:")
        print("1. Kaggle Dogs vs Cats: https://www.kaggle.com/c/dogs-vs-cats/data")
        print("2. 或使用 download_data.py 腳本下載範例資料")
        return False

    return True


def get_data_loaders():
    """準備訓練和驗證資料載入器"""

    # 訓練資料轉換 (包含資料增強)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),           # 隨機水平翻轉
        transforms.RandomRotation(15),               # 隨機旋轉
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 色彩抖動
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet 均值
                             [0.229, 0.224, 0.225])  # ImageNet 標準差
    ])

    # 驗證資料轉換 (不做資料增強)
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 載入資料集
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)

    # 分割訓練集和驗證集
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    # 為驗證集設定不同的轉換
    val_dataset.dataset.transform = val_transform

    # 建立資料載入器
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"\n訓練資料數量: {len(train_dataset)}")
    print(f"驗證資料數量: {len(val_dataset)}")
    print(f"類別: {full_dataset.classes}")

    return train_loader, val_loader, full_dataset.classes


# ============== 模型定義 ==============
class SimpleCNN(nn.Module):
    """自定義簡單 CNN 模型"""

    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_pretrained_model(num_classes=2):
    """取得預訓練模型 (ResNet18)"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 凍結預訓練層 (可選)
    for param in model.parameters():
        param.requires_grad = False

    # 替換最後一層分類器
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    return model


# ============== 訓練函數 ==============
def train_one_epoch(model, train_loader, optimizer, criterion, epoch):
    """訓練一個 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for data, target in pbar:
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, val_loader, criterion):
    """評估模型"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


# ============== 視覺化 ==============
def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """繪製訓練歷史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs_range = range(1, len(train_losses) + 1)

    # 損失曲線
    ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs_range, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # 準確率曲線
    ax2.plot(epochs_range, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs_range, val_accs, 'r-', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    print("訓練歷史已儲存至 training_history.png")


def visualize_predictions(val_loader, model, classes):
    """視覺化預測結果"""
    model.eval()

    # 取一批資料
    data, target = next(iter(val_loader))
    data, target = data.to(DEVICE), target.to(DEVICE)

    with torch.no_grad():
        output = model(data)
        _, predicted = output.max(1)

    # 反正規化用於顯示
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # 繪製前 8 張
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        if i >= len(data):
            break

        img = data[i].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = img.clip(0, 1)

        ax.imshow(img)

        pred_label = classes[predicted[i]]
        true_label = classes[target[i]]
        color = 'green' if predicted[i] == target[i] else 'red'

        ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}', color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_samples.png', dpi=150)
    plt.show()
    print("預測範例已儲存至 prediction_samples.png")


# ============== 主程式 ==============
def main():
    print("=" * 50)
    print("貓狗分類器 - CNN 訓練")
    print("=" * 50)
    print(f"使用裝置: {DEVICE}")
    print(f"使用預訓練模型: {USE_PRETRAINED}")
    print()

    # 確保模型目錄存在
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 檢查資料
    print("[1] 檢查資料目錄...")
    if not setup_data_directory():
        return
    print()

    # 準備資料
    print("[2] 準備資料載入器...")
    train_loader, val_loader, classes = get_data_loaders()
    print()

    # 建立模型
    print("[3] 建立模型...")
    if USE_PRETRAINED:
        model = get_pretrained_model(num_classes=len(classes)).to(DEVICE)
        print("使用預訓練 ResNet18 模型")
    else:
        model = SimpleCNN(num_classes=len(classes)).to(DEVICE)
        print("使用自定義 CNN 模型")
    print()

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()

    if USE_PRETRAINED:
        # 只優化最後一層
        optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 學習率調度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 訓練歷史紀錄
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0

    # 開始訓練
    print("[4] 開始訓練...")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        # 訓練
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch
        )

        # 評估
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # 更新學習率
        scheduler.step()

        # 紀錄
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"         Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # 儲存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': classes,
                'val_acc': val_acc,
            }, MODEL_SAVE_PATH)
            print(f"         >> 儲存最佳模型 (Val Acc: {val_acc:.2f}%)")

        print()

    print("-" * 50)
    print()

    # 視覺化
    print("[5] 視覺化結果...")
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    visualize_predictions(val_loader, model, classes)

    print()
    print("=" * 50)
    print("訓練完成!")
    print(f"最佳驗證準確率: {best_val_acc:.2f}%")
    print(f"模型已儲存至: {MODEL_SAVE_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()
