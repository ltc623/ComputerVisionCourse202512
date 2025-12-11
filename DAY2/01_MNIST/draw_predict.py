"""
MNIST 手寫數字辨識 - 滑鼠手寫畫布
使用滑鼠在畫布上書寫數字，即時辨識

操作說明:
- 按住滑鼠左鍵書寫數字
- 按 'c' 清除畫布
- 按 'p' 進行預測
- 按 'q' 退出程式
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import os

# ============== 取得腳本所在目錄 ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============== 設定區 ==============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "mnist_cnn.pth")

# 畫布設定
CANVAS_SIZE = 400       # 畫布大小
BRUSH_SIZE = 20         # 筆刷大小
BRUSH_COLOR = 255       # 筆刷顏色 (白色)


# ============== 模型定義 (與訓練相同) ==============
class SimpleCNN(nn.Module):
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


# ============== 全域變數 ==============
canvas = None
drawing = False
last_point = None
model = None


# ============== 滑鼠回調函數 ==============
def mouse_callback(event, x, y, flags, param):
    global canvas, drawing, last_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
        cv2.circle(canvas, (x, y), BRUSH_SIZE // 2, BRUSH_COLOR, -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, last_point, (x, y), BRUSH_COLOR, BRUSH_SIZE)
            last_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_point = None


# ============== 載入模型 ==============
def load_model(model_path):
    model = SimpleCNN().to(DEVICE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"找不到模型檔案: {model_path}\n"
            "請先執行 train.py 訓練模型"
        )

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


# ============== 影像預處理 ==============
def preprocess_canvas(canvas_img):
    """
    將畫布影像預處理成 MNIST 格式
    """
    # 找輪廓定位數字
    contours, _ = cv2.findContours(canvas_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 找最大輪廓
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 加入邊距
        margin = 30
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(canvas_img.shape[1] - x, w + 2 * margin)
        h = min(canvas_img.shape[0] - y, h + 2 * margin)

        digit_roi = canvas_img[y:y+h, x:x+w]

        # 置中到正方形
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square[y_offset:y_offset+h, x_offset:x_offset+w] = digit_roi

        # 調整大小 (MNIST 風格: 20x20 數字 + 4 像素邊距)
        resized = cv2.resize(square, (20, 20), interpolation=cv2.INTER_AREA)
        final = np.zeros((28, 28), dtype=np.uint8)
        final[4:24, 4:24] = resized
    else:
        # 直接縮放
        final = cv2.resize(canvas_img, (28, 28), interpolation=cv2.INTER_AREA)

    return final


def image_to_tensor(image):
    """轉換為模型輸入張量"""
    normalized = image.astype(np.float32) / 255.0
    normalized = (normalized - 0.1307) / 0.3081
    tensor = torch.from_numpy(normalized).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor


# ============== 預測 ==============
def predict(model, image_tensor):
    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item(), probabilities.squeeze().cpu().numpy()


# ============== 建立顯示畫面 ==============
def create_display(canvas_img, processed_img=None, prediction=None, confidence=None, probs=None):
    """建立完整顯示畫面"""

    # 主畫布 (轉 BGR 顯示)
    display = cv2.cvtColor(canvas_img, cv2.COLOR_GRAY2BGR)

    # 繪製邊框
    cv2.rectangle(display, (0, 0), (CANVAS_SIZE-1, CANVAS_SIZE-1), (0, 255, 0), 2)

    # 操作說明
    info_panel = np.zeros((CANVAS_SIZE, 250, 3), dtype=np.uint8)

    instructions = [
        "=== Instructions ===",
        "",
        "Draw with mouse",
        "",
        "[C] Clear canvas",
        "[P] Predict",
        "[Q] Quit",
        "",
        "===================",
    ]

    for i, text in enumerate(instructions):
        cv2.putText(info_panel, text, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 顯示預測結果
    if prediction is not None:
        result_y = 280
        cv2.putText(info_panel, "=== Result ===", (10, result_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.putText(info_panel, f"Digit: {prediction}", (10, result_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.putText(info_panel, f"Conf: {confidence*100:.1f}%", (10, result_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 機率條
        if probs is not None:
            bar_y_start = result_y + 90
            bar_width = 100
            bar_height = 12

            for i, prob in enumerate(probs):
                y = bar_y_start + i * (bar_height + 3)

                # 背景
                cv2.rectangle(info_panel, (50, y), (50 + bar_width, y + bar_height),
                              (80, 80, 80), -1)

                # 機率條
                prob_w = int(bar_width * prob)
                color = (0, 255, 0) if i == prediction else (150, 150, 150)
                cv2.rectangle(info_panel, (50, y), (50 + prob_w, y + bar_height),
                              color, -1)

                # 標籤
                cv2.putText(info_panel, str(i), (30, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # 顯示 28x28 預覽
    if processed_img is not None:
        preview_size = 112
        preview = cv2.resize(processed_img, (preview_size, preview_size),
                             interpolation=cv2.INTER_NEAREST)
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        # 放在 info panel 右上角
        preview_x = 130
        preview_y = 30
        info_panel[preview_y:preview_y+preview_size,
                   preview_x:preview_x+preview_size] = preview_bgr

        cv2.putText(info_panel, "28x28", (preview_x + 30, preview_y + preview_size + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # 合併畫面
    combined = np.hstack([display, info_panel])

    return combined


# ============== 主程式 ==============
def main():
    global canvas, model

    print("=" * 50)
    print("MNIST 滑鼠手寫辨識")
    print("=" * 50)
    print(f"使用裝置: {DEVICE}")
    print()

    # 載入模型
    print("[1] 載入模型...")
    try:
        model = load_model(MODEL_PATH)
        print(f"模型已從 {MODEL_PATH} 載入")
    except FileNotFoundError as e:
        print(f"錯誤: {e}")
        return
    print()

    # 初始化畫布 (黑底)
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

    # 建立視窗
    window_name = "MNIST Draw & Predict"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("[2] 畫布已就緒")
    print("-" * 50)
    print("操作說明:")
    print("  - 按住滑鼠左鍵書寫數字")
    print("  - 按 'c' 清除畫布")
    print("  - 按 'p' 進行預測")
    print("  - 按 'q' 退出")
    print("-" * 50)

    # 預測結果暫存
    processed_img = None
    prediction = None
    confidence = None
    probs = None

    while True:
        # 建立顯示畫面
        display = create_display(canvas, processed_img, prediction, confidence, probs)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n退出程式")
            break

        elif key == ord('c'):
            # 清除畫布
            canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
            processed_img = None
            prediction = None
            confidence = None
            probs = None
            print("畫布已清除")

        elif key == ord('p'):
            # 檢查是否有畫內容
            if np.sum(canvas) == 0:
                print("畫布是空的，請先書寫數字")
                continue

            # 預處理
            processed_img = preprocess_canvas(canvas)

            # 轉換為張量
            tensor = image_to_tensor(processed_img)

            # 預測
            prediction, confidence, probs = predict(model, tensor)

            print(f"辨識結果: {prediction} (信心度: {confidence*100:.1f}%)")

    cv2.destroyAllWindows()
    print("\n程式結束")


if __name__ == "__main__":
    main()
