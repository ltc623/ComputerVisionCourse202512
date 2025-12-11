"""
MNIST 手寫數字辨識 - 訓練腳本
使用 PyTorch 建立簡單的 CNN 模型進行 0-9 數字分類
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt

# ============== 取得腳本所在目錄 (相對路徑基準) ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============== 設定區 ==============
BATCH_SIZE = 64          # 批次大小
EPOCHS = 20              # 訓練輪數
LEARNING_RATE = 0.001    # 學習率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 相對路徑設定
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "mnist_cnn.pth")

# ============== 資料準備 ==============
def get_data_loaders():
    """準備訓練和測試資料載入器"""

    # 資料轉換：轉為 Tensor 並正規化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和標準差
    ])

    # 下載並載入 MNIST 訓練資料
    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    # 下載並載入 MNIST 測試資料
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    # 建立資料載入器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"訓練資料數量: {len(train_dataset)}")
    print(f"測試資料數量: {len(test_dataset)}")

    return train_loader, test_loader


# ============== 模型定義 ==============
class SimpleCNN(nn.Module):
    """簡單的 CNN 模型"""

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 卷積層 1: 輸入 1 通道 (灰階), 輸出 32 通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 卷積層 2: 輸入 32 通道, 輸出 64 通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 池化層
        self.pool = nn.MaxPool2d(2, 2)

        # 全連接層
        # 輸入: 64 * 7 * 7 (經過兩次池化後 28->14->7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 個類別 (0-9)

        # Dropout 防止過擬合
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 卷積層 1 + ReLU + 池化
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))

        # 卷積層 2 + ReLU + 池化
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))

        # 展平
        x = x.view(-1, 64 * 7 * 7)

        # 全連接層
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


# ============== 訓練函數 ==============
def train_one_epoch(model, train_loader, optimizer, criterion, epoch):
    """訓練一個 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        # 清除梯度
        optimizer.zero_grad()

        # 前向傳播
        output = model(data)

        # 計算損失
        loss = criterion(output, target)

        # 反向傳播
        loss.backward()

        # 更新權重
        optimizer.step()

        # 統計
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # 每 100 個 batch 顯示一次進度
        if batch_idx % 100 == 0:
            print(f"  Batch [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion):
    """評估模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


# ============== 視覺化 ==============
def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """繪製訓練歷史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 損失曲線
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(test_losses, label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)

    # 準確率曲線
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()
    print("訓練歷史已儲存至 training_history.png")


def visualize_samples(test_loader, model):
    """視覺化預測結果"""
    model.eval()

    # 取一批資料
    data, target = next(iter(test_loader))
    data, target = data.to(DEVICE), target.to(DEVICE)

    with torch.no_grad():
        output = model(data)
        _, predicted = output.max(1)

    # 繪製前 16 張
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        img = data[i].cpu().squeeze().numpy()
        ax.imshow(img, cmap='gray')

        color = 'green' if predicted[i] == target[i] else 'red'
        ax.set_title(f'Pred: {predicted[i].item()} / True: {target[i].item()}',
                     color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_samples.png', dpi=150)
    plt.show()
    print("預測範例已儲存至 prediction_samples.png")


# ============== 主程式 ==============
def main():
    print("=" * 50)
    print("MNIST 手寫數字辨識 - CNN 訓練")
    print("=" * 50)
    print(f"使用裝置: {DEVICE}")
    print()

    # 確保模型目錄存在
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 準備資料
    print("[1] 準備資料...")
    train_loader, test_loader = get_data_loaders()
    print()

    # 建立模型
    print("[2] 建立模型...")
    model = SimpleCNN().to(DEVICE)
    print(model)
    print()

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 訓練歷史紀錄
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    # 開始訓練
    print("[3] 開始訓練...")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # 訓練
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch
        )

        # 評估
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        # 紀錄
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")

    print("-" * 50)
    print()

    # 儲存模型
    print("[4] 儲存模型...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_accs': test_accs,
    }, MODEL_SAVE_PATH)
    print(f"模型已儲存至 {MODEL_SAVE_PATH}")
    print()

    # 視覺化
    print("[5] 視覺化結果...")
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    visualize_samples(test_loader, model)

    print()
    print("=" * 50)
    print("訓練完成!")
    print(f"最終測試準確率: {test_accs[-1]:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
