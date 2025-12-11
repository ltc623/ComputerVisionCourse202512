"""
MNIST 手寫數字辨識 - WebCam 即時辨識
使用 OpenCV 擷取攝影機畫面，即時辨識手寫數字

操作說明:
1. 將手寫數字紙張對準畫面中央的綠色框
2. 按 'c' 擷取並辨識
3. 按 'q' 退出程式

注意:
- 建議使用深色筆在白紙上書寫
- 確保光線充足，避免陰影
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import os

# ============== 取得腳本所在目錄 ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============== 設定區 ==============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "mnist_cnn.pth")

# ROI (感興趣區域) 設定
ROI_SIZE = 280  # 擷取區域大小 (正方形)


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


# ============== 載入模型 ==============
def load_model(model_path):
    """載入訓練好的模型"""
    model = SimpleCNN().to(DEVICE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"找不到模型檔案: {model_path}\n"
            "請先執行 train.py 訓練模型"
        )

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型已從 {model_path} 載入")
    return model


# ============== 影像預處理 ==============
def preprocess_for_mnist(roi_image):
    """
    將擷取的 ROI 影像預處理成 MNIST 格式

    MNIST 特性:
    - 28x28 像素
    - 灰階
    - 白底黑字 -> 黑底白字 (需反轉)
    - 數字置中
    """
    # 轉灰階
    if len(roi_image.shape) == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_image.copy()

    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 自適應二值化 (處理不均勻光線)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,  # 反轉: 白紙黑字 -> 黑底白字
        11, 2
    )

    # 找輪廓以定位數字
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 找最大輪廓 (假設是數字)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 擷取數字區域並加入邊距
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(binary.shape[1] - x, w + 2 * margin)
        h = min(binary.shape[0] - y, h + 2 * margin)

        digit_roi = binary[y:y+h, x:x+w]

        # 將數字區域置中到正方形
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)

        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = digit_roi

        # 調整大小到 28x28，保留邊距 (MNIST 風格)
        # MNIST 數字通常佔據 20x20 區域，周圍有 4 像素邊距
        resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)

        # 加入邊距成為 28x28
        final = np.zeros((28, 28), dtype=np.uint8)
        final[4:24, 4:24] = resized

    else:
        # 沒有找到輪廓，直接縮放
        final = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)

    return final, binary


def image_to_tensor(image):
    """將預處理後的影像轉換為模型輸入張量"""
    # 正規化到 [0, 1]
    normalized = image.astype(np.float32) / 255.0

    # 套用 MNIST 的標準化
    normalized = (normalized - 0.1307) / 0.3081

    # 轉為 PyTorch 張量
    tensor = torch.from_numpy(normalized).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]

    return tensor


# ============== 預測 ==============
def predict(model, image_tensor):
    """進行預測"""
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item(), probabilities.squeeze().cpu().numpy()


# ============== 繪製 UI ==============
def draw_ui(frame, roi_rect, prediction=None, confidence=None, probs=None):
    """繪製使用者介面"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi_rect

    # 繪製 ROI 框 (綠色)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 繪製操作說明
    instructions = [
        "Press 'c' to Capture & Recognize",
        "Press 'q' to Quit",
        "Place digit in green box"
    ]

    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 顯示預測結果
    if prediction is not None:
        # 結果背景
        result_y = h - 150
        cv2.rectangle(frame, (10, result_y), (200, h - 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, result_y), (200, h - 10), (0, 255, 0), 2)

        # 預測數字
        cv2.putText(frame, f"Digit: {prediction}", (20, result_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # 信心度
        cv2.putText(frame, f"Conf: {confidence*100:.1f}%", (20, result_y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 繪製機率條
        if probs is not None:
            bar_x = w - 150
            bar_width = 100
            bar_height = 15

            cv2.putText(frame, "Probabilities:", (bar_x - 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for i, prob in enumerate(probs):
                y = 50 + i * (bar_height + 5)

                # 背景條
                cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height),
                              (100, 100, 100), -1)

                # 機率條
                prob_width = int(bar_width * prob)
                color = (0, 255, 0) if i == prediction else (200, 200, 200)
                cv2.rectangle(frame, (bar_x, y), (bar_x + prob_width, y + bar_height),
                              color, -1)

                # 數字標籤
                cv2.putText(frame, str(i), (bar_x - 20, y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame


def draw_processed_preview(frame, processed_img, binary_img, roi_rect):
    """在畫面上顯示處理後的預覽"""
    x1, y1, x2, y2 = roi_rect

    # 顯示預處理後的 28x28 影像 (放大顯示)
    preview_size = 112  # 28 * 4
    preview = cv2.resize(processed_img, (preview_size, preview_size),
                         interpolation=cv2.INTER_NEAREST)
    preview_color = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    # 放在 ROI 框的右邊
    preview_x = x2 + 20
    preview_y = y1

    if preview_x + preview_size < frame.shape[1]:
        frame[preview_y:preview_y+preview_size,
              preview_x:preview_x+preview_size] = preview_color

        cv2.putText(frame, "28x28 Input", (preview_x, preview_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return frame


# ============== 主程式 ==============
def main():
    print("=" * 50)
    print("MNIST WebCam 即時手寫數字辨識")
    print("=" * 50)
    print(f"使用裝置: {DEVICE}")
    print()

    # 載入模型
    print("[1] 載入模型...")
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        return
    print()

    # 開啟攝影機
    print("[2] 開啟攝影機...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("錯誤: 無法開啟攝影機")
        print("請確認:")
        print("  1. 攝影機已連接")
        print("  2. 沒有其他程式正在使用攝影機")
        return

    # 取得攝影機解析度
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"攝影機解析度: {frame_width}x{frame_height}")

    # 計算 ROI 位置 (置中)
    roi_x1 = (frame_width - ROI_SIZE) // 2
    roi_y1 = (frame_height - ROI_SIZE) // 2
    roi_x2 = roi_x1 + ROI_SIZE
    roi_y2 = roi_y1 + ROI_SIZE
    roi_rect = (roi_x1, roi_y1, roi_x2, roi_y2)

    print()
    print("[3] 開始即時辨識...")
    print("-" * 50)
    print("操作說明:")
    print("  - 將手寫數字對準綠色框")
    print("  - 按 'c' 擷取並辨識")
    print("  - 按 'q' 退出")
    print("-" * 50)

    # 預測結果暫存
    last_prediction = None
    last_confidence = None
    last_probs = None
    last_processed = None
    last_binary = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("警告: 無法讀取攝影機畫面")
            break

        # 如需鏡像模式，取消下行註解
        # frame = cv2.flip(frame, 1)

        # 擷取 ROI
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()

        # 繪製 UI
        display_frame = draw_ui(frame, roi_rect,
                                last_prediction, last_confidence, last_probs)

        # 顯示處理預覽
        if last_processed is not None:
            display_frame = draw_processed_preview(display_frame, last_processed,
                                                   last_binary, roi_rect)

        # 顯示畫面
        cv2.imshow("MNIST WebCam Recognition", display_frame)

        # 鍵盤輸入
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n退出程式")
            break

        elif key == ord('c'):
            print("\n擷取畫面...")

            # 預處理
            processed, binary = preprocess_for_mnist(roi)
            last_processed = processed
            last_binary = binary

            # 轉換為張量
            tensor = image_to_tensor(processed)

            # 預測
            pred, conf, probs = predict(model, tensor)
            last_prediction = pred
            last_confidence = conf
            last_probs = probs

            print(f"辨識結果: {pred} (信心度: {conf*100:.1f}%)")

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()
    print("\n程式結束")


if __name__ == "__main__":
    main()
