import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import joblib
from GetDataLoader import GetDataLoader
from Modle import FeatureExtractor
import json
# ===== 配置參數 =====
# MODEL_NAME = "IDEA-CCNL/Erlangshen-RoBERTa-110M-Sentiment"
MODEL_NAME = "intfloat/multilingual-e5-large"
with open('HW3/key.json', 'r') as f:
    keys = json.load(f)
USE_AUTH_TOKEN = keys.get('HW3_Key')
# TRAIN_PATH = r'C:\Users\H514 #4856\Desktop\deep_learning_114206103\HW3_Data\split_data\train_data.csv'
# VAL_PATH = r'C:\Users\H514 #4856\Desktop\deep_learning_114206103\HW3_Data\split_data\val_data.csv'
# TEST_PATH = r'C:\Users\H514 #4856\Desktop\deep_learning_114206103\HW3_Data\split_data\test_data.csv'
TRAIN_PATH = r'C:\Users\H514 #4856\Desktop\deep_learning_114206103\HW3_Data\online_shopping\train_data.csv'  # 訓練資料路徑
TEST_PATH = r'C:\Users\H514 #4856\Desktop\deep_learning_114206103\HW3_Data\online_shopping\test_data.csv'    # 測試資料路徑
VAL_PATH = None  # 驗證資料路徑 (None 表示從訓練資料中分割)

BATCH_SIZE = 200
NUM_CLASSES = 2

# SVC 參數
SVC_CONFIGS = [
    # {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
    # {'C': 10.0, 'kernel': 'rbf', 'gamma': 'scale'},
    # {'C': 1.0, 'kernel': 'rbf'},
    # {'C': 10.0, 'kernel': 'rbf'},
    {'C': 10.0, 'kernel': 'linear'},
    {'C': 1.0, 'kernel': 'linear'},
]

# 實驗結果資料夾
EXPERIMENTS_DIR = 'HW3/svc_experiments'
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def extract_features(feature_extractor, dataloader, device, desc="Extracting features"):
    """
    使用特徵提取器提取所有樣本的特徵
    """
    feature_extractor.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            # 提取特徵
            features = feature_extractor(input_ids, attention_mask)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels)
    
    # 合併所有批次
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    return all_features, all_labels


def plot_confusion_matrix(y_true, y_pred, save_dir, class_names=['Negative', 'Positive']):
    """繪製混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix (SVC)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩陣已儲存至 {save_path}")
    plt.close()


def main():
    print("="*60)
    print("SVC 情緒分類模型訓練")
    print("使用大型語言模型作為特徵提取器")
    print("="*60)
    
    # 1. 載入資料
    print("\n[1/5] 載入資料...")
    data_loader = GetDataLoader(MODEL_NAME, use_auth_token=USE_AUTH_TOKEN)
    train_dataloader, val_dataloader, test_dataloader = data_loader.get_DataLoader(
        data_sum=-1,
        batch_size=BATCH_SIZE,
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        test_path=TEST_PATH
    )
    print(f"訓練集批次數: {len(train_dataloader)}")
    print(f"驗證集批次數: {len(val_dataloader)}")
    print(f"測試集批次數: {len(test_dataloader)}")
    
    # 2. 初始化特徵提取器
    print("\n[2/5] 初始化特徵提取器...")
    feature_extractor = FeatureExtractor(MODEL_NAME, use_auth_token=USE_AUTH_TOKEN)
    
    # 調整 tokenizer
    if data_loader.tokenizer is not None:
        feature_extractor.model.resize_token_embeddings(len(data_loader.tokenizer))
    
    feature_extractor.to(device)
    print(f"特徵維度: {feature_extractor.model.config.hidden_size}")
    
    # 3. 提取特徵
    print("\n[3/5] 提取所有資料的特徵...")
    X_train, y_train = extract_features(feature_extractor, train_dataloader, device, "提取訓練集特徵")
    X_val, y_val = extract_features(feature_extractor, val_dataloader, device, "提取驗證集特徵")
    X_test, y_test = extract_features(feature_extractor, test_dataloader, device, "提取測試集特徵")
    
    print(f"訓練集特徵形狀: {X_train.shape}")
    print(f"驗證集特徵形狀: {X_val.shape}")
    print(f"測試集特徵形狀: {X_test.shape}")
    
    # 4. 標準化特徵
    print("\n[4/5] 標準化特徵...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. 訓練 SVC 模型（多個配置）
    print("\n[5/5] 訓練 SVC 模型...")
    
    results = []
    
    for idx, svc_config in enumerate(SVC_CONFIGS, 1):
        print(f"\n{'='*60}")
        print(f"實驗 {idx}/{len(SVC_CONFIGS)}")
        print(f"配置: {svc_config}")
        print(f"{'='*60}")
        
        # 創建實驗資料夾
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        kernel = svc_config['kernel']
        C = svc_config['C']
        exp_dir = os.path.join(EXPERIMENTS_DIR, f"exp_{timestamp}_svc_{kernel}_C{C}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # 訓練 SVC
        print("\n訓練 SVC...")
        svc = SVC(**svc_config, random_state=42, verbose=True)
        svc.fit(X_train_scaled, y_train)
        
        # 驗證集評估
        print("\n驗證集評估...")
        y_val_pred = svc.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_precision = precision_score(y_val, y_val_pred, average='weighted')
        val_recall = recall_score(y_val, y_val_pred, average='weighted')
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        print(f"\n驗證集結果:")
        print(f"  準確度: {val_accuracy:.4f}")
        print(f"  精確度: {val_precision:.4f}")
        print(f"  召回率: {val_recall:.4f}")
        print(f"  F1分數: {val_f1:.4f}")
        
        # 測試集評估
        print("\n測試集評估...")
        y_test_pred = svc.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        print(f"\n測試集結果:")
        print(f"  準確度: {test_accuracy:.4f}")
        print(f"  精確度: {test_precision:.4f}")
        print(f"  召回率: {test_recall:.4f}")
        print(f"  F1分數: {test_f1:.4f}")
        
        # 繪製混淆矩陣
        plot_confusion_matrix(y_test, y_test_pred, exp_dir)
        
        # 保存模型
        model_path = os.path.join(exp_dir, 'svc_model.pkl')
        scaler_path = os.path.join(exp_dir, 'scaler.pkl')
        joblib.dump(svc, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"\n模型已儲存至: {model_path}")
        print(f"Scaler已儲存至: {scaler_path}")
        
        # 保存詳細報告
        report = classification_report(y_test, y_test_pred, target_names=['Negative', 'Positive'])
        report_path = os.path.join(exp_dir, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("SVC 情緒分類實驗報告\n")
            f.write("="*60 + "\n\n")
            f.write(f"實驗時間: {timestamp}\n")
            f.write(f"特徵提取器: {MODEL_NAME}\n")
            f.write(f"SVC 配置: {svc_config}\n\n")
            f.write("驗證集結果:\n")
            f.write(f"  準確度: {val_accuracy:.4f}\n")
            f.write(f"  精確度: {val_precision:.4f}\n")
            f.write(f"  召回率: {val_recall:.4f}\n")
            f.write(f"  F1分數: {val_f1:.4f}\n\n")
            f.write("測試集結果:\n")
            f.write(f"  準確度: {test_accuracy:.4f}\n")
            f.write(f"  精確度: {test_precision:.4f}\n")
            f.write(f"  召回率: {test_recall:.4f}\n")
            f.write(f"  F1分數: {test_f1:.4f}\n\n")
            f.write("詳細分類報告:\n")
            f.write(report)
        
        print(f"報告已儲存至: {report_path}")
        
        # 記錄結果
        results.append({
            '實驗時間': timestamp,
            '特徵提取器': MODEL_NAME,
            'Kernel': kernel,
            'C': C,
            'Gamma': svc_config.get('gamma', 'N/A'),
            '驗證準確度': val_accuracy,
            '驗證精確度': val_precision,
            '驗證召回率': val_recall,
            '驗證F1': val_f1,
            '測試準確度': test_accuracy,
            '測試精確度': test_precision,
            '測試召回率': test_recall,
            '測試F1': test_f1,
        })
    
    # 6. 保存所有實驗結果
    print("\n" + "="*60)
    print("所有實驗完成！")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    results_path = os.path.join(EXPERIMENTS_DIR, 'svc_experiments_summary.csv')
    
    # 如果檔案已存在，則附加（append）；否則創建新檔案
    if os.path.exists(results_path):
        # 使用 mode='a' 直接附加到檔案尾端，不會覆蓋原有資料
        results_df.to_csv(results_path, mode='a', header=False, index=False, encoding='utf-8-sig')
        print(f"\n實驗結果已附加至: {results_path}")
    else:
        # 檔案不存在，創建新檔案並寫入標題
        results_df.to_csv(results_path, mode='w', header=True, index=False, encoding='utf-8-sig')
        print(f"\n實驗摘要已儲存至: {results_path}")
    
    # 顯示最佳結果（從本次實驗中）
    best_idx = results_df['測試準確度'].idxmax()
    best_result = results_df.iloc[best_idx]
    print("\n🏆 本次實驗最佳配置:")
    print(f"  Kernel: {best_result['Kernel']}")
    print(f"  C: {best_result['C']}")
    print(f"  測試準確度: {best_result['測試準確度']:.4f}")
    print(f"  測試F1: {best_result['測試F1']:.4f}")


if __name__ == '__main__':
    main()
