import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm # 用於顯示漂亮的進度條
from Model import SimpleCNN, load_datasets, draw_train_loss, compute_and_save_metrics, draw_confusion_matrix 
import multiprocessing
try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False
try:
    from sklearn.metrics import confusion_matrix as _sk_confusion_matrix
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False
import numpy as np
from PIL import Image
import shutil

# --- 2. 偵測裝置 (GPU 或 CPU) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)
# --- 2. 設定資料路徑 ---
# ！！【請務必修改】！！
# 這是你建立的 "Corn_Leaves_Split" 資料夾的「父目錄」路徑
# 預設為 'Corn_Leaves_Split'，若該資料夾位於 `HW2/Corn_Leaves_Split`，腳本會自動嘗試尋找。
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(BASE_DIR, "output_data_augmented")  # 修改此行

# 如果上面指定的路徑找不到，嘗試幾個常見候選路徑（相對於專案根或本檔案目錄）
if not os.path.isdir(BASE_DATA_DIR):
    # 優先嘗試專案子目錄 HW2/Corn_Leaves_Split
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, "output_data_augmented")
    if os.path.isdir(candidate):
        BASE_DATA_DIR = candidate
    else:
        # 再嘗試以當前工作目錄為基準
        candidate2 = os.path.join(os.getcwd(), BASE_DATA_DIR)
        if os.path.isdir(candidate2):
            BASE_DATA_DIR = candidate2

# --- 訓練參數 ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 100
NUM_CLASSES = 4        
LEARNING_RATE = 0.001  # 學習率
NUM_EPOCHS = 35        

# 儲存最佳模型的檔案名稱
BEST_MODEL_PATH = "./HW2/corn_leaf_best_model.pth"

# --- 3. 定義資料轉換 (Transforms) ---

# (1) 訓練集 (train) 的轉換 -> 包含資料增強
train_transforms = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)), # 縮放
    # T.RandomHorizontalFlip(),          # 隨機水平翻轉
    # T.RandomRotation(15),              # 隨機旋轉 10 度
    T.ToTensor(),                      # 轉換為 Tensor (0.0 ~ 1.0)
    T.Normalize(mean=[0.485, 0.456, 0.406], # 標準化
                std=[0.229, 0.224, 0.225])
])

# (2) 測試集/驗證集 (test) 的轉換 -> *不*包含資料增強
test_transforms = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)), # 只做縮放
    T.ToTensor(),                      # 轉換為 Tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], # 標準化
                std=[0.229, 0.224, 0.225])
])


def main():
    # 載入資料集
    try:
        train_loader, val_loader, test_loader, class_names = load_datasets(
            data_dir=BASE_DATA_DIR,
            train_transform=train_transforms,
            eval_transform=test_transforms,
            batch_size=BATCH_SIZE,
            num_workers=4
        )
        # 用於計算 dataset 長度
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        test_dataset = test_loader.dataset
    except Exception as e:
        print(f"\n【錯誤！】載入資料時發生錯誤：{e}")
        return

    # --- 6. 建立模型、損失函數、優化器 ---

    print("\n--- 正在建立模型 ---")
    model = SimpleCNN(num_classes=NUM_CLASSES)
    model = model.to(DEVICE) # ！！將模型移動到 GPU/CPU！！
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("模型建立完畢，並已移動到 " + str(DEVICE))

    # --- 7. 執行訓練迴圈 ---

    print(f"\n--- 即將開始訓練，共 {NUM_EPOCHS} 回合 (Epochs) ---")

    # 用於儲存最好的模型
    best_val_accuracy = 0.0
    # 用於繪圖的損失記錄
    train_losses = []
    val_losses = []
    # 迴圈跑 NUM_EPOCHS 次
    for epoch in range(NUM_EPOCHS):
        
        # --- (A) 訓練階段 (Training) ---
        model.train() # ！！將模型切換到「訓練模式」！！
        running_train_loss = 0.0
        
        # 使用 tqdm 顯示進度條
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")

        for images, labels in train_progress_bar:
            # 1. 將資料移動到 DEVICE
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # 2. 梯度歸零
            optimizer.zero_grad()
            
            # 3. 前向傳播 (預測)
            outputs = model(images)
            
            # 4. 計算損失
            loss = criterion(outputs, labels)
            
            # 5. 反向傳播 (計算梯度)
            loss.backward()
            
            # 6. 更新權重
            optimizer.step()
            
            # 累加損失
            running_train_loss += loss.item() * images.size(0)
        
        # 計算此 Epoch 的平均訓練損失
        epoch_train_loss = running_train_loss / len(train_dataset)

        # --- (B) 驗證階段 (Validation / Testing) ---
        model.eval() # ！！將模型切換到「評估模式」！！
        
        running_val_loss = 0.0
        correct_predictions = 0

        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]")

        # 關閉梯度計算，節省記憶體並加速
        with torch.no_grad():
            for images, labels in val_progress_bar:
                # 1. 將資料移動到 DEVICE
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                # 2. 前向傳播
                outputs = model(images)
                # 3. 計算損失
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                # 4. 計算準確度
                _, predicted_labels = torch.max(outputs, 1)
                correct_predictions += (predicted_labels == labels).sum().item()

        # 計算此 Epoch 的平均驗證損失和總準確度
        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_accuracy = correct_predictions / len(val_dataset)

        # 記錄損失用於後續繪圖
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # --- (C) 印出結果 ---
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} 完成:")
        print(f"  [Train] Loss: {epoch_train_loss:.4f}")
        print(f"  [Val]   Loss: {epoch_val_loss:.4f} | Accuracy: {epoch_val_accuracy:.4f} ({(epoch_val_accuracy*100):.2f}%)")

        # --- (D) 儲存最好的模型 ---
        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            # 儲存模型的「狀態字典」(state_dict)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> 新高！驗證準確度達到 {(best_val_accuracy*100):.2f}%。模型已儲存至 {BEST_MODEL_PATH}")

    print("\n--- 訓練完成 ---")
    print(f"最好的驗證準確度為: {(best_val_accuracy*100):.2f}%")
    print(f"最好的模型已儲存在 {BEST_MODEL_PATH}")

    # 最終在測試集上評估
    print('\n--- 在測試集上做最終評估 ---')
    test_running_loss = 0.0
    test_correct = 0
    all_true = []
    all_pred = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Final Test'):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            # collect for confusion matrix
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    test_loss = test_running_loss / len(test_dataset)
    test_acc = test_correct / len(test_dataset)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # 計算並儲存 Accuracy/Precision/Recall/F1
    try:
        compute_and_save_metrics(all_true, all_pred, class_names=class_names, out_csv='Expansion_metrics.csv')
    except Exception as e:
        print(f"產生評估指標時發生錯誤: {e}")

    # 繪製並儲存訓練/驗證損失曲線
    draw_train_loss(train_losses, val_losses, out_path='Expansion_train_val_loss.png')

    # 繪製並儲存混淆矩陣（原始與正規化）
    try:
        draw_confusion_matrix(all_true, all_pred, class_names=class_names, out_path='Expansion_confusion_matrix.png', normalize=False)
        draw_confusion_matrix(all_true, all_pred, class_names=class_names, out_path='Expansion_confusion_matrix_normalized.png', normalize=True)
    except Exception as e:
        print(f"產生混淆矩陣時發生錯誤: {e}")

if __name__ == '__main__':
    # 在 Windows 上啟動多程序前先呼叫 freeze_support
    multiprocessing.freeze_support()
    main()