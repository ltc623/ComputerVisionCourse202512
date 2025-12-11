"""
MNIST 手寫數字辨識 - 預測腳本
載入訓練好的模型對新影像進行預測
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

# ============== 取得腳本所在目錄 ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============== 設定區 ==============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "mnist_cnn.pth")


# ============== 模型定義 (與訓練相同) ==============
class SimpleCNN(nn.Module):
    """簡單的 CNN 模型"""

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ============== 預處理 ==============
def preprocess_image(image_path):
    """
    預處理輸入影像
    - 轉為灰階
    - 調整大小至 28x28
    - 正規化
    """
    # 定義轉換
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 載入影像
    image = Image.open(image_path)

    # 轉換
    image_tensor = transform(image)

    # 加入批次維度
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


# ============== 載入模型 ==============
def load_model(model_path):
    """載入訓練好的模型"""
    model = SimpleCNN().to(DEVICE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔案: {model_path}\n請先執行 train.py 訓練模型")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型已從 {model_path} 載入")
    return model


# ============== 預測 ==============
def predict(model, image_tensor):
    """進行預測"""
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)

        # 取得機率 (softmax)
        probabilities = torch.softmax(output, dim=1)

        # 取得預測結果
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item(), probabilities.squeeze().cpu().numpy()


# ============== 視覺化 ==============
def visualize_prediction(image_path, predicted, confidence, probabilities):
    """視覺化預測結果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 顯示原始影像
    image = Image.open(image_path).convert('L')
    ax1.imshow(image, cmap='gray')
    ax1.set_title(f'Prediction: {predicted}\nConfidence: {confidence*100:.2f}%',
                  fontsize=14)
    ax1.axis('off')

    # 顯示各類別機率
    classes = list(range(10))
    colors = ['green' if i == predicted else 'steelblue' for i in classes]

    ax2.barh(classes, probabilities, color=colors)
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Digit')
    ax2.set_title('Class Probabilities')
    ax2.set_yticks(classes)
    ax2.set_xlim(0, 1)

    for i, (prob, cls) in enumerate(zip(probabilities, classes)):
        ax2.text(prob + 0.02, cls, f'{prob*100:.1f}%', va='center')

    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150)
    plt.show()
    print("預測結果已儲存至 prediction_result.png")


# ============== 主程式 ==============
def main():
    parser = argparse.ArgumentParser(description='MNIST 數字辨識預測')
    parser.add_argument('--image', '-i', type=str, required=True,
                        help='輸入影像路徑')
    parser.add_argument('--model', '-m', type=str, default=MODEL_PATH,
                        help='模型檔案路徑')

    args = parser.parse_args()

    print("=" * 50)
    print("MNIST 手寫數字辨識 - 預測")
    print("=" * 50)
    print(f"使用裝置: {DEVICE}")
    print(f"輸入影像: {args.image}")
    print()

    # 檢查影像是否存在
    if not os.path.exists(args.image):
        print(f"錯誤: 找不到影像檔案 {args.image}")
        return

    # 載入模型
    print("[1] 載入模型...")
    model = load_model(args.model)
    print()

    # 預處理影像
    print("[2] 預處理影像...")
    image_tensor = preprocess_image(args.image)
    print(f"影像張量形狀: {image_tensor.shape}")
    print()

    # 進行預測
    print("[3] 進行預測...")
    predicted, confidence, probabilities = predict(model, image_tensor)
    print()

    # 顯示結果
    print("=" * 50)
    print(f"預測結果: {predicted}")
    print(f"信心度:   {confidence*100:.2f}%")
    print("=" * 50)

    # 視覺化
    print()
    print("[4] 視覺化結果...")
    visualize_prediction(args.image, predicted, confidence, probabilities)


if __name__ == "__main__":
    main()
