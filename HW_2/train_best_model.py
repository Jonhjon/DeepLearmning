"""
使用最佳超參數配置的模型
根據超參數搜尋結果: SimpleCNN2_deeper, kernel_size=3, padding=1, stride=1
進行 5 次獨立訓練並生成綜合報告
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
import numpy as np
import csv
from datetime import datetime

# 導入自定義函式
from Model import load_datasets, draw_train_loss, compute_and_save_metrics, draw_confusion_matrix

# --- 偵測裝置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 

# --- 設定資料路徑 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DATA_DIR = os.path.join(BASE_DIR, "output_data_extra_augmented")

if not os.path.isdir(BASE_DATA_DIR):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, "output_data_extra_augmented")
    if os.path.isdir(candidate):
        BASE_DATA_DIR = candidate
    else:
        candidate2 = os.path.join(os.getcwd(), "HW2", "output_data_extra_augmented")
        if os.path.isdir(candidate2):
            BASE_DATA_DIR = candidate2

# --- 訓練參數 ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 90  # 增加 batch size 提高 GPU 利用率
NUM_WORKERS = 8  # 增加數據載入執行緒數
NUM_CLASSES = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 100  # 使用最佳配置可以訓練更多 epochs
NUM_RUNS = 5  # 訓練次數

# === 最佳超參數配置 (來自超參數搜尋結果) ===
BEST_KERNEL_SIZE = 3
BEST_PADDING = 1
BEST_STRIDE = 1

# 儲存路徑
OUTPUT_DIR = "./HW2/best_model_5runs_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 資料轉換 ---
train_transforms = T.Compose([
    # T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    # T.RandomHorizontalFlip(p=0.5),
    # T.RandomRotation(15),
    # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = T.Compose([
    # T.Resize((IMG_HEIGHT, IMG_WIDTH)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === 使用最佳超參數的模型 ===
class OptimizedCNN(nn.Module):
    """
    基於超參數搜尋結果的最佳模型
    架構: SimpleCNN2_deeper
    超參數: kernel_size=3, padding=1, stride=1
    驗證準確度: 97.44%
    測試準確度: 93.51%
    """
    def __init__(self, num_classes=4):
        super().__init__()
        
        # 使用最佳超參數
        kernel_size = BEST_KERNEL_SIZE
        padding = BEST_PADDING
        stride = BEST_STRIDE
        
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=kernel_size, padding=padding, stride=stride), 
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding, stride=stride), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding, stride=stride), 
            nn.BatchNorm2d(128), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=kernel_size, padding=padding, stride=stride), 
            nn.BatchNorm2d(256), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 5: 256 -> 512
            nn.Conv2d(256, 512, kernel_size=kernel_size, padding=padding, stride=stride), 
            nn.BatchNorm2d(512), 
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_single_run(run_id, train_loader, val_loader, test_loader, class_names, train_dataset, val_dataset, test_dataset):
    """執行單次訓練"""
    print(f"\n{'='*70}")
    print(f"第 {run_id}/{NUM_RUNS} 次訓練")
    print(f"{'='*70}\n")
    
    # 設定隨機種子以確保可重現性（每次不同）
    torch.manual_seed(42 + run_id)
    np.random.seed(42 + run_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42 + run_id)
    
    # 創建模型
    model = OptimizedCNN(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 訓練變數
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    run_dir = os.path.join(OUTPUT_DIR, "all_runs", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    best_model_path = os.path.join(run_dir, f"model_run_{run_id}.pth")

    # 訓練迴圈
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
            optimizer.step()
            
            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
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
        
        # 調整學習率
        scheduler.step(epoch_val_loss)
        
        # 顯示結果
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} 完成:")
        print(f"  [Train] Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f} ({epoch_train_acc*100:.2f}%)")
        print(f"  [Val]   Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f} ({epoch_val_acc*100:.2f}%)\n")
        
        # 儲存最佳模型
        if epoch_val_acc > best_val_accuracy:
            best_val_accuracy = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
        if (epoch + 1) % 10 == 0:
            print(f"  ✓ 新高！Val Acc: {best_val_accuracy*100:.2f}%")

    print(f"\n第 {run_id} 次訓練完成! 最佳驗證準確度: {best_val_accuracy*100:.2f}%")

    # --- 測試階段 ---
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
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
    
    print(f"測試準確度: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # 儲存結果
    prefix = f"run_{run_id}"
    
    compute_and_save_metrics(
        all_true, all_pred, 
        class_names=class_names, 
        out_csv=os.path.join(run_dir, f'{prefix}_metrics.csv')
    )
    
    draw_train_loss(
        train_losses, val_losses, 
        out_path=os.path.join(run_dir, f'{prefix}_loss.png')
    )
    
    draw_confusion_matrix(
        all_true, all_pred, 
        class_names=class_names, 
        out_path=os.path.join(run_dir, f'{prefix}_confusion_matrix.png'), 
        normalize=False
    )
    
    return {
        'run_id': run_id,
        'best_val_acc': best_val_accuracy,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'all_true': all_true,
        'all_pred': all_pred
    }


def generate_comprehensive_report(all_results, class_names, total_params, trainable_params):
    """生成綜合報告"""
    print(f"\n{'='*70}")
    print("生成綜合報告")
    print(f"{'='*70}\n")
    
    # 計算統計數據
    val_accs = [r['best_val_acc'] for r in all_results]
    test_accs = [r['test_acc'] for r in all_results]
    test_losses = [r['test_loss'] for r in all_results]
    
    mean_val_acc = np.mean(val_accs)
    std_val_acc = np.std(val_accs)
    mean_test_acc = np.mean(test_accs)
    std_test_acc = np.std(test_accs)
    mean_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)
    
    # 儲存統計 CSV
    stats_csv = os.path.join(OUTPUT_DIR, "statistics_summary.csv")
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Run', 'Val_Accuracy', 'Test_Accuracy', 'Test_Loss'])
        for r in all_results:
            writer.writerow([
                r['run_id'],
                f"{r['best_val_acc']:.4f}",
                f"{r['test_acc']:.4f}",
                f"{r['test_loss']:.4f}"
            ])
        writer.writerow([])
        writer.writerow(['Metric', 'Mean', 'Std', 'Min', 'Max'])
        writer.writerow(['Val_Accuracy', f"{mean_val_acc:.4f}", f"{std_val_acc:.4f}", 
                        f"{min(val_accs):.4f}", f"{max(val_accs):.4f}"])
        writer.writerow(['Test_Accuracy', f"{mean_test_acc:.4f}", f"{std_test_acc:.4f}", 
                        f"{min(test_accs):.4f}", f"{max(test_accs):.4f}"])
        writer.writerow(['Test_Loss', f"{mean_test_loss:.4f}", f"{std_test_loss:.4f}", 
                        f"{min(test_losses):.4f}", f"{max(test_losses):.4f}"])
    
    # 生成文字報告
    report_path = os.path.join(OUTPUT_DIR, "comprehensive_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"訓練綜合報告 - {NUM_RUNS} 次獨立訓練\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("模型配置:\n")
        f.write("  架構: SimpleCNN2_deeper (5層卷積)\n")
        f.write(f"  Kernel Size: {BEST_KERNEL_SIZE}x{BEST_KERNEL_SIZE}\n")
        f.write(f"  Padding: {BEST_PADDING}\n")
        f.write(f"  Stride: {BEST_STRIDE}\n")
        f.write(f"  總參數量: {total_params:,}\n")
        f.write(f"  可訓練參數: {trainable_params:,}\n\n")
        
        f.write("訓練設定:\n")
        f.write(f"  Learning Rate: {LEARNING_RATE}\n")
        f.write(f"  Batch Size: {BATCH_SIZE}\n")
        f.write(f"  Epochs: {NUM_EPOCHS}\n")
        f.write(f"  訓練次數: {NUM_RUNS}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("統計結果\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"驗證準確度:\n")
        f.write(f"  平均: {mean_val_acc:.4f} ({mean_val_acc*100:.2f}%)\n")
        f.write(f"  標準差: {std_val_acc:.4f} ({std_val_acc*100:.2f}%)\n")
        f.write(f"  最小值: {min(val_accs):.4f} ({min(val_accs)*100:.2f}%)\n")
        f.write(f"  最大值: {max(val_accs):.4f} ({max(val_accs)*100:.2f}%)\n\n")
        
        f.write(f"測試準確度:\n")
        f.write(f"  平均: {mean_test_acc:.4f} ({mean_test_acc*100:.2f}%)\n")
        f.write(f"  標準差: {std_test_acc:.4f} ({std_test_acc*100:.2f}%)\n")
        f.write(f"  最小值: {min(test_accs):.4f} ({min(test_accs)*100:.2f}%)\n")
        f.write(f"  最大值: {max(test_accs):.4f} ({max(test_accs)*100:.2f}%)\n\n")
        
        f.write(f"測試損失:\n")
        f.write(f"  平均: {mean_test_loss:.4f}\n")
        f.write(f"  標準差: {std_test_loss:.4f}\n")
        f.write(f"  最小值: {min(test_losses):.4f}\n")
        f.write(f"  最大值: {max(test_losses):.4f}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("各次訓練詳細結果\n")
        f.write("=" * 70 + "\n\n")
        
        for r in all_results:
            f.write(f"第 {r['run_id']} 次訓練:\n")
            f.write(f"  驗證準確度: {r['best_val_acc']:.4f} ({r['best_val_acc']*100:.2f}%)\n")
            f.write(f"  測試準確度: {r['test_acc']:.4f} ({r['test_acc']*100:.2f}%)\n")
            f.write(f"  測試損失: {r['test_loss']:.4f}\n\n")
    
    # 顯示摘要
    print(f"{'='*70}")
    print(f"訓練完成! 共進行 {NUM_RUNS} 次獨立訓練")
    print(f"{'='*70}\n")
    print(f"驗證準確度: {mean_val_acc:.4f} ± {std_val_acc:.4f} ({mean_val_acc*100:.2f}% ± {std_val_acc*100:.2f}%)")
    print(f"測試準確度: {mean_test_acc:.4f} ± {std_test_acc:.4f} ({mean_test_acc*100:.2f}% ± {std_test_acc*100:.2f}%)")
    print(f"測試損失: {mean_test_loss:.4f} ± {std_test_loss:.4f}")
    print(f"\n報告已儲存:")
    print(f"  - 統計摘要: {stats_csv}")
    print(f"  - 完整報告: {report_path}")
    print(f"  - 各次結果: {OUTPUT_DIR}/run_1/ ~ run_{NUM_RUNS}/")
    print(f"{'='*70}\n")


def main():
    print(f"\n{'='*70}")
    print(f"使用最佳超參數配置進行 {NUM_RUNS} 次訓練")
    print(f"{'='*70}")
    print(f"模型架構: SimpleCNN2_deeper (5層卷積)")
    print(f"最佳超參數:")
    print(f"  - Kernel Size: {BEST_KERNEL_SIZE}x{BEST_KERNEL_SIZE}")
    print(f"  - Padding: {BEST_PADDING}")
    print(f"  - Stride: {BEST_STRIDE}")
    print(f"  - Using device: {DEVICE}")
    print(f"訓練設定:")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - 訓練次數: {NUM_RUNS}")
    print(f"{'='*70}\n")
    
    # 載入資料集（只需載入一次）
    try:
        train_loader, val_loader, test_loader, class_names = load_datasets(
            data_dir=BASE_DATA_DIR,
            train_transform=train_transforms,
            eval_transform=test_transforms,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        test_dataset = test_loader.dataset
    except Exception as e:
        print(f"\n【錯誤！】載入資料時發生錯誤：{e}")
        return
    
    # 取得模型參數資訊
    temp_model = OptimizedCNN(num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in temp_model.parameters())
    trainable_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    del temp_model
    
    print(f"模型資訊:")
    print(f"  總參數量: {total_params:,}")
    print(f"  可訓練參數: {trainable_params:,}\n")
    
    # 執行多次訓練
    all_results = []
    for run_id in range(1, NUM_RUNS + 1):
        result = train_single_run(
            run_id, train_loader, val_loader, test_loader, 
            class_names, train_dataset, val_dataset, test_dataset
        )
        all_results.append(result)
    
    # 生成綜合報告
    generate_comprehensive_report(all_results, class_names, total_params, trainable_params)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
