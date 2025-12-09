import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from Modle import MLP
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Num GPUs Available:", torch.cuda.device_count())

path = r'C:\Users\H514 #4856\Desktop\deep learning 114206103\HW1\archive/dataset.csv'

binary_cols =['Gender','Daytime_evening_attendance','Displaced','Educational special needs','Debtor'
              ,'Tuition fees up to date','Scholarship holder','International']

categorical_cols = ['Marital status','Application mode','Course'
                    ,'Previous qualification','Nacionality','Mother qualification'
                    ,'Father qualification','Mother occupation'
                    ,'Father occupation']

numeric_cols =['Application order','Age at enrollment','Curricular units 1st sem (credited)'
               ,'Curricular units 1st sem (enrolled)','Curricular units 1st sem (evaluations)',
               'Curricular units 1st sem (approved)','Curricular units 1st sem (grade)',
               'Curricular units 1st sem (without evaluations)',
               'Curricular units 2nd sem (credited)','Curricular units 2nd sem (enrolled)',
               'Curricular units 2nd sem (evaluations)','Curricular units 2nd sem (approved)',
               'Curricular units 2nd sem (grade)','Curricular units 2nd sem (without evaluations)',
               'Unemployment rate','Inflation rate','GDP']

df = pd.read_csv(path)
print(f"Original DataFrame shape: {df.shape}")

# 將 feature 列為 numeric（非數值會被轉為 NaN → 填 0）
X_df = df.drop('Target', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)

# Label encode target（用於相關係數計算與後續模型）
le = LabelEncoder()
y = le.fit_transform(df['Target'].values)
y_ser = pd.Series(y, index=X_df.index)

# 計算各特徵與 target 的 Pearson correlation（取絕對值並排序）
corr_with_target = X_df.corrwith(y_ser).abs().sort_values(ascending=False)

# 參數
top_n = 30
corr_thresh = 0.90
os.makedirs('data_analyze', exist_ok=True)

# 繪製與 target 相關性最高的前 top_n 特徵之 pairwise correlation heatmap
top_feats = list(corr_with_target.head(top_n).index)
plt.figure(figsize=(min(16, len(top_feats)), min(12, len(top_feats))))
sns.heatmap(X_df[top_feats].corr(), cmap='vlag', center=0, linewidths=0.5)
plt.title(f'Heatmap of top {len(top_feats)} features by |corr| with target')
plt.tight_layout()
plt.savefig(os.path.join('data_analyze', f'top_{len(top_feats)}_feat_corr_heatmap.png'))
plt.close()

# 貪婪式相關性篩選：按與 target 的相關性高到低選特徵，若與已選特徵 pairwise corr > thresh 則捨棄
kept = []
for feat in corr_with_target.index:
    keep = True
    for k in kept:
        if abs(X_df[feat].corr(X_df[k])) > corr_thresh:
            keep = False
            break
    if keep:
        kept.append(feat)

print(f"Original feature count: {X_df.shape[1]}, selected feature count: {len(kept)}")
with open(os.path.join('data_analyze', 'selected_features_by_corr.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(kept))

# 使用篩選後的欄位作為後續處理的起點（保留 DataFrame）
X_selected = X_df[kept].copy()

# 只處理實際存在的欄位（避免 KeyError）
binary_exist = [c for c in binary_cols if c in X_selected.columns]
cat_exist = [c for c in categorical_cols if c in X_selected.columns]
num_exist = [c for c in numeric_cols if c in X_selected.columns]

# 將 binary 欄位轉為字串再做 get_dummies（避免 0/1 被當作數值）
if binary_exist:
    X_selected[binary_exist] = X_selected[binary_exist].astype(str)

# One-hot 處理 binary + categorical（只處理存在的欄位）
onehot_cols = binary_exist + cat_exist
if onehot_cols:
    X_selected = pd.get_dummies(X_selected, columns=onehot_cols, drop_first=True)

# 對數值欄位做 Min-Max 正規化（只處理仍存在的數值欄位）
num_exist_after = [c for c in num_exist if c in X_selected.columns]
if num_exist_after:
    scaler = MinMaxScaler()
    X_selected[num_exist_after] = scaler.fit_transform(X_selected[num_exist_after])

print(f"Transformed DataFrame shape (after encoding/scaling): {X_selected.shape}")

# 轉為 numpy 供模型使用
X = X_selected.values.astype(np.float32)
# y 已由 LabelEncoder 轉成數值
print(f"Final feature matrix shape: {X.shape}, label vector shape: {y.shape}")
# print(y)  # (4209, 57) (4209, 56) (4209,)
# 從訓練的CSV中分出訓練跟測試7:3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 從訓練資料中再分出訓練跟驗證8:2
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

# 轉成 tensor
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.long)
# X_val = torch.tensor(X_val, dtype=torch.float32)
# y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 建立 DataLoader
# train_dataset = TensorDataset(X_train, y_train)
# val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
# train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 設定參數
n_splits = 5
num_epochs = 1000
lr= 0.001
task = 'best'
print(X.shape[1], len(np.unique(y)))

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
best_models = []

best_train_loss = []
best_val_loss = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):

    model = MLP(intput_dim=X.shape[1], num_classes=len(np.unique(y))).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adadelta(model.parameters(), lr=lr)
    optimizer = optim.Adadelta(model.parameters())

    print(f"\n--- Fold {fold}/{n_splits} ---")
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # 建立 DataLoader
    train_dataset = TensorDataset(X_tr, y_tr)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)


    train_loss = []
    train_acc = []
    val_loss_list = []

    best_val_acc_fold = -1.0
    best_state_fold = None
    best_epoch = -1

    # 訓練模型
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        # 紀錄訓練損失
        train_loss.append(loss.item())
        train_acc.append((outputs.argmax(dim=1) == batch_y).float().mean().item())

        # 驗證模型
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)
        # val_f1 = f1_score(val_labels, val_preds, average='weighted')
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))

        if val_acc > best_val_acc_fold:
            best_val_acc_fold = val_acc
            best_state_fold = {k:v.cpu() for k,v in model.state_dict().items()}
            best_epoch = epoch + 1
            best_train_loss = train_loss.copy()
            best_val_loss = val_loss_list.copy()

        print(f"Epoch {epoch+1}/{num_epochs}, train_loss : {train_loss[-1]:.4f}, train_acc : {train_acc[-1]:.4f} , Val Loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")
    if best_state_fold is not None:
        best_models.append((best_val_acc_fold, best_state_fold, fold))

if len(best_models) == 0:
    raise RuntimeError("No trained models saved from folds.")

best_val_acc, best_state, best_fold = max(best_models, key=lambda x: x[0])
print(f"\nSelected best fold: {best_fold} with val acc: {best_val_acc:.4f}")

base = os.path.join(r'C:\Users\H514 #4856\Desktop\deep learning 114206103\\HW1', task)
# os.makedirs(base, exist_ok=True)

if not os.path.exists(base):
    os.makedirs(base)
#畫出best_train_loss和best_val_loss的圖
plt.figure(figsize=(10, 6))
plt.plot(best_train_loss, label='Train Loss')
plt.plot(best_val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss (Best Fold: {best_fold})')
plt.legend()
plt.grid()
plt.savefig(os.path.join(base,f"loss_curve_best_fold_{best_fold}_feature.jpg"))

# 在測試集上評估模型
best_model = MLP(intput_dim=X.shape[1], num_classes=len(np.unique(y))).to(device)
best_model.load_state_dict(best_state)

model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        test_labels.extend(batch_y.cpu().numpy())

test_precision = precision_score(y_test.numpy(), test_preds, average='weighted')
print(classification_report(y_test.numpy(), test_preds))
test_recall = recall_score(y_test.numpy(), test_preds, average='weighted')
test_f1 = f1_score(y_test.numpy(), test_preds, average='weighted')
test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
print(f"(best model from fold {best_fold})\n"
      f"Test Accuracy : {test_acc:.4f},\n"
      f"Test Precision : {test_precision:.4f},\n"
      f"Test Recall: {test_recall:.4f},\n"
      f"Test F1 Score: {test_f1:.4f}\n")
