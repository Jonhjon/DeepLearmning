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

# (1) 對 0/1 類別型做 One-Hot
df[binary_cols] = df[binary_cols].astype(str)
df = pd.get_dummies(df, columns=binary_cols, drop_first=True)

#對一般類別型欄位做 One-Hot
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# (3) 對數值型欄位做 Min-Max 正規化
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(f"Transformed DataFrame shape: {df.shape}")

X_df = df.drop('Target', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0)
X = X_df.values.astype(np.float32)
y = df['Target'].values
# print(X.shape)
print(f"Transformed DataFrame shape: {df.shape}")

le = LabelEncoder()
y = le.fit_transform(y)
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
task = 'different_optim'
print(X.shape[1], len(np.unique(y)))

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
best_models = []

best_train_loss = []
best_val_loss = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):

    model = MLP(intput_dim=X.shape[1], num_classes=len(np.unique(y))).to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adadelta(model.parameters(), lr=lr)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

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
plt.savefig(os.path.join(base,f"loss_curve_best_fold_{best_fold}_Adadelta.jpg"))

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
