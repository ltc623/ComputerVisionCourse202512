"""
貓狗分類器 - 預測腳本
載入訓練好的模型對新影像進行預測
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

# ============== 取得腳本所在目錄 ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============== 設定區 ==============
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "catdog_model.pth")


# ============== 模型定義 ==============
def get_model(num_classes=2):
    """取得與訓練相同結構的模型"""
    model = models.resnet18(weights=None)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )

    return model


# ============== 預處理 ==============
def preprocess_image(image_path):
    """預處理輸入影像"""
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


# ============== 載入模型 ==============
def load_model(model_path):
    """載入訓練好的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"找不到模型檔案: {model_path}\n請先執行 train.py 訓練模型"
        )

    checkpoint = torch.load(model_path, map_location=DEVICE)
    classes = checkpoint['classes']

    model = get_model(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型已從 {model_path} 載入")
    print(f"類別: {classes}")

    return model, classes


# ============== 預測 ==============
def predict(model, image_tensor, classes):
    """進行預測"""
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = classes[predicted.item()]
    confidence_value = confidence.item()
    probs = probabilities.squeeze().cpu().numpy()

    return predicted_class, confidence_value, probs


# ============== 視覺化 ==============
def visualize_prediction(image_path, predicted_class, confidence, probs, classes):
    """視覺化預測結果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # 顯示原始影像
    image = Image.open(image_path).convert('RGB')
    ax1.imshow(image)
    ax1.set_title(f'Prediction: {predicted_class}\nConfidence: {confidence*100:.2f}%',
                  fontsize=14)
    ax1.axis('off')

    # 顯示各類別機率
    colors = ['green' if cls == predicted_class else 'steelblue' for cls in classes]

    bars = ax2.barh(classes, probs, color=colors)
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Class')
    ax2.set_title('Class Probabilities')
    ax2.set_xlim(0, 1)

    for bar, prob in zip(bars, probs):
        ax2.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{prob*100:.1f}%', va='center')

    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150)
    plt.show()
    print("預測結果已儲存至 prediction_result.png")


# ============== 批次預測 ==============
def predict_batch(model, image_paths, classes):
    """批次預測多張影像"""
    results = []

    for path in image_paths:
        if not os.path.exists(path):
            print(f"警告: 找不到檔案 {path}, 跳過")
            continue

        image_tensor = preprocess_image(path)
        pred_class, conf, _ = predict(model, image_tensor, classes)
        results.append({
            'path': path,
            'prediction': pred_class,
            'confidence': conf
        })

    return results


# ============== 主程式 ==============
def main():
    parser = argparse.ArgumentParser(description='貓狗分類器預測')
    parser.add_argument('--image', '-i', type=str,
                        help='單張輸入影像路徑')
    parser.add_argument('--folder', '-f', type=str,
                        help='批次預測資料夾路徑')
    parser.add_argument('--model', '-m', type=str, default=MODEL_PATH,
                        help='模型檔案路徑')

    args = parser.parse_args()

    if not args.image and not args.folder:
        parser.print_help()
        print("\n請提供 --image 或 --folder 參數")
        return

    print("=" * 50)
    print("貓狗分類器 - 預測")
    print("=" * 50)
    print(f"使用裝置: {DEVICE}")
    print()

    # 載入模型
    print("[1] 載入模型...")
    model, classes = load_model(args.model)
    print()

    if args.image:
        # 單張預測
        print(f"[2] 預測影像: {args.image}")

        if not os.path.exists(args.image):
            print(f"錯誤: 找不到影像檔案 {args.image}")
            return

        image_tensor = preprocess_image(args.image)
        pred_class, conf, probs = predict(model, image_tensor, classes)

        print()
        print("=" * 50)
        print(f"預測結果: {pred_class}")
        print(f"信心度:   {conf*100:.2f}%")
        print("=" * 50)

        print()
        print("[3] 視覺化結果...")
        visualize_prediction(args.image, pred_class, conf, probs, classes)

    elif args.folder:
        # 批次預測
        print(f"[2] 批次預測資料夾: {args.folder}")

        if not os.path.exists(args.folder):
            print(f"錯誤: 找不到資料夾 {args.folder}")
            return

        # 找出所有圖片
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_paths = [
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if f.lower().endswith(image_extensions)
        ]

        if not image_paths:
            print("錯誤: 資料夾中沒有找到圖片")
            return

        print(f"找到 {len(image_paths)} 張圖片")
        print()

        results = predict_batch(model, image_paths, classes)

        print("=" * 50)
        print("預測結果:")
        print("-" * 50)

        cat_count = 0
        dog_count = 0

        for r in results:
            filename = os.path.basename(r['path'])
            print(f"  {filename}: {r['prediction']} ({r['confidence']*100:.1f}%)")

            if r['prediction'] == 'cats':
                cat_count += 1
            else:
                dog_count += 1

        print("-" * 50)
        print(f"統計: 貓 {cat_count} 張, 狗 {dog_count} 張")
        print("=" * 50)


if __name__ == "__main__":
    main()
