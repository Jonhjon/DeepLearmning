"""
使用預訓練 Vision Transformer (ViT) 模型進行玉米葉病害分類
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
import numpy as np

# 導入預訓練模型
import timm  # PyTorch Image Models 函式庫

# 導入自定義函式
from Model import load_datasets, draw_train_loss, compute_and_save_metrics, draw_confusion_matrix

# --- 偵測裝置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 設定資料路徑 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(BASE_DIR, "output_data_augmented")

if not os.path.isdir(BASE_DATA_DIR):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, "output_data_augmented")
    if os.path.isdir(candidate):
        BASE_DATA_DIR = candidate
    else:
        candidate2 = os.path.join(os.getcwd(), "HW2", "output_data_augmented")
        if os.path.isdir(candidate2):
            BASE_DATA_DIR = candidate2

# --- 訓練參數 ---
IMG_HEIGHT = 224  # ViT 標準輸入大小
IMG_WIDTH = 224
BATCH_SIZE = 64  # ViT 建議使用較大的 batch size
NUM_CLASSES = 4
LEARNING_RATE = 5e-5  # 使用預訓練模型時建議較小的學習率
NUM_EPOCHS = 40
WARMUP_EPOCHS = 3  # 學習率預熱

#輸出我的訓練參數

# 模型選擇
# ResNet 系列: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
# ViT 系列: 'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224'
MODEL_NAME = 'vit_base_patch16_224'

# 儲存路徑
BEST_MODEL_PATH = f"./HW2/{MODEL_NAME}_best_model.pth"
OUTPUT_PREFIX = f"{MODEL_NAME}_AdamW_L2"

# --- 資料轉換 ---
# ViT 使用 ImageNet 的標準化參數
train_transforms = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def create_model(model_name, num_classes, pretrained=True):
    """
    創建預訓練模型 (支援 ResNet 和 ViT)
    
    參數:
        model_name: 模型名稱
        num_classes: 分類類別數
        pretrained: 是否使用預訓練權重
    """
    print(f"\n{'='*70}")
    print(f"創建模型: {model_name}")
    print(f"{'='*70}")
    
    # 使用 timm 創建模型
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    # 顯示模型資訊
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型參數總數: {total_params:,}")
    print(f"可訓練參數: {trainable_params:,}")
    print(f"使用預訓練權重: {pretrained}")
    print(f"{'='*70}\n")
    
    return model

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    學習率調度器: 預熱 + 餘弦退火
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # 預熱階段: 線性增加
            return float(current_step) / float(max(1, num_warmup_steps))
        # 餘弦退火
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    # 顯示設備資訊
    print(f"\n{'='*70}")
    print("Vision Transformer 訓練程式")
    print(f"{'='*70}")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*70}\n")
    
    # --- 載入資料集 ---
    try:
        train_loader, val_loader, test_loader, class_names = load_datasets(
            data_dir=BASE_DATA_DIR,
            train_transform=train_transforms,
            eval_transform=test_transforms,
            batch_size=BATCH_SIZE,
            num_workers=4
        )
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        test_dataset = test_loader.dataset
    except Exception as e:
        print(f"\n【錯誤！】載入資料時發生錯誤：{e}")
        return

    # --- 創建模型 ---
    model = create_model(MODEL_NAME, NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)

    # --- 損失函數和優化器 ---
    criterion = nn.CrossEntropyLoss()
    
    # 使用 AdamW 優化器 (Transformer 推薦)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01  # L2 正則化
    )
    
    # 學習率調度器
    num_training_steps = NUM_EPOCHS * len(train_loader)
    num_warmup_steps = WARMUP_EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    print(f"\n{'='*70}")
    print(f"開始訓練 - 共 {NUM_EPOCHS} 個 Epochs")
    print(f"{'='*70}\n")

    # --- 訓練迴圈 ---
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []

    for epoch in range(NUM_EPOCHS):
        # === 訓練階段 ===
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for images, labels in train_pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # 每個 batch 更新學習率
            
            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            
            # 更新進度條
            current_lr = scheduler.get_last_lr()[0]
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.6f}'
            })
        
        epoch_train_loss = running_train_loss / len(train_dataset)
        epoch_train_acc = train_correct / len(train_dataset)
        
        # === 驗證階段 ===
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
        
        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_acc = val_correct / len(val_dataset)
        
        # 記錄
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accuracies.append(epoch_train_acc)
        val_accuracies.append(epoch_val_acc)
        learning_rates.append(scheduler.get_last_lr()[0])
        
        # 顯示結果
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} 完成:")
        print(f"  [Train] Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f} ({epoch_train_acc*100:.2f}%)")
        print(f"  [Val]   Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f} ({epoch_val_acc*100:.2f}%)")
        print(f"  Learning Rate: {learning_rates[-1]:.6f}")
        
        # 儲存最佳模型
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✓ 新高！已儲存模型 (Val Acc: {best_val_accuracy*100:.2f}%)")

    print(f"\n{'='*70}")
    print("訓練完成!")
    print(f"{'='*70}")
    print(f"最佳驗證準確度: {best_val_accuracy*100:.2f}%")
    print(f"模型已儲存至: {BEST_MODEL_PATH}")

    # --- 測試階段 ---
    print(f"\n{'='*70}")
    print("載入最佳模型進行測試")
    print(f"{'='*70}\n")
    
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=False))
    model.eval()
    
    test_running_loss = 0.0
    test_correct = 0
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
    
    test_loss = test_running_loss / len(test_dataset)
    test_acc = test_correct / len(test_dataset)
    
    print(f"\n測試結果:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # --- 儲存結果 ---
    print(f"\n{'='*70}")
    print("生成評估報告")
    print(f"{'='*70}\n")
    
    # 計算並儲存指標
    try:
        compute_and_save_metrics(
            all_true, all_pred, 
            class_names=class_names, 
            out_csv=f'{OUTPUT_PREFIX}_metrics.csv'
        )
    except Exception as e:
        print(f"生成評估指標時發生錯誤: {e}")

    # 繪製損失曲線
    try:
        draw_train_loss(
            train_losses, val_losses, 
            out_path=f'{OUTPUT_PREFIX}_loss.png'
        )
    except Exception as e:
        print(f"繪製損失曲線時發生錯誤: {e}")

    # 繪製混淆矩陣
    try:
        draw_confusion_matrix(
            all_true, all_pred, 
            class_names=class_names, 
            out_path=f'{OUTPUT_PREFIX}_confusion_matrix.png', 
            normalize=False
        )
        draw_confusion_matrix(
            all_true, all_pred, 
            class_names=class_names, 
            out_path=f'{OUTPUT_PREFIX}_confusion_matrix_normalized.png', 
            normalize=True
        )
    except Exception as e:
        print(f"繪製混淆矩陣時發生錯誤: {e}")

    print(f"\n{'='*70}")
    print("所有處理完成!")
    print(f"{'='*70}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
