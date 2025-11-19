import os
import shutil
import random
import splitfolders

# 來源與目標目錄（相對於此腳本）
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = os.path.join(BASE_DIR, 'data_256')
# OUT_DIR = os.path.join(BASE_DIR, 'Corn_Leaves_Split')

# # 切分比例
# TEST_PCT = 0.15
# VAL_PCT = 0.15
# TRAIN_PCT = 0.70

# # 支援的影像副檔名
# IMG_EXTS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}

# random.seed(42)

# if not os.path.isdir(SRC_DIR):
#     print(f"來源資料夾不存在: {SRC_DIR}")
#     raise SystemExit(1)

# os.makedirs(OUT_DIR, exist_ok=True)

# classes = [d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d)) and not d.startswith('__')]
# if not classes:
#     print(f"在 {SRC_DIR} 未找到任何 class 子資料夾")
#     raise SystemExit(1)

# summary = {}
# for cls in classes:
#     cls_src = os.path.join(SRC_DIR, cls)
#     files = [f for f in os.listdir(cls_src) if os.path.splitext(f.lower())[1] in IMG_EXTS]
#     files = [os.path.join(cls_src, f) for f in files]
#     random.shuffle(files)
#     total = len(files)
#     test_count = int(total * TEST_PCT)
#     val_count = int(total * VAL_PCT)
#     train_count = total - test_count - val_count

#     train_files = files[:train_count]
#     val_files = files[train_count:train_count + val_count]
#     test_files = files[train_count + val_count:train_count + val_count + test_count]

#     # create target dirs
#     for split_name, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
#         out_split_dir = os.path.join(OUT_DIR, split_name, cls)
#         os.makedirs(out_split_dir, exist_ok=True)
#         for src_path in split_files:
#             dst_path = os.path.join(out_split_dir, os.path.basename(src_path))
#             shutil.copy2(src_path, dst_path)

#     summary[cls] = (train_count, val_count, test_count)

# print('切分完成，輸出資料夾:', OUT_DIR)
# for cls, counts in summary.items():
#     print(f"  {cls}: train={counts[0]}, val={counts[1]}, test={counts[2]}")

# # 顯示總計
# tot_train = sum(v[0] for v in summary.values())
# tot_val = sum(v[1] for v in summary.values())
# tot_test = sum(v[2] for v in summary.values())
# print(f"總計 -> train={tot_train}, val={tot_val}, test={tot_test}")
import splitfolders
import os

# --- 設定路徑 ---
# 輸入資料夾的路徑 (你的原始資料)
# 注意：因為你的路徑中有特殊符號和空格，建議使用 r"..." 來表示原始字串
input_folder = r"C:\Users\H514 #4856\Desktop\deep_learning_114206103\HW2\data_256"

# 輸出資料夾的路徑 (切分後的資料要存去哪裡)
output_folder = "output_data"

# --- 執行切分 ---
# ratio 參數分別代表 (訓練集, 驗證集, 測試集)
# seed 參數設為固定數字 (例如 42)，確保每次執行結果都一樣
print("正在開始切分資料，請稍候...")

splitfolders.ratio(
    input_folder, 
    output=output_folder, 
    seed=42, 
    ratio=(0.7, 0.15, 0.15), 
    group_prefix=None, 
    move=False # False 代表複製檔案 (保留原檔)，True 代表移動檔案
)

print(f"資料切分完成！請查看資料夾: {output_folder}")
