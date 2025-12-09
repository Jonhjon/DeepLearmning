"""
對訓練資料進行預處理
對每張圖片應用: RandomHorizontalFlip + RandomRotation + ColorJitter
不增加圖片數量，只是對現有圖片進行轉換
"""
import os
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# --- 設定路徑 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "output_data_augmented")
OUTPUT_DIR = os.path.join(BASE_DIR, "output_data_extra_augmented")

# --- 資料轉換 ---
# 對圖片進行轉換處理
process_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
])


def process_single_image(img_path, output_path):
    """對單張圖片進行轉換處理"""
    try:
        # 讀取圖片
        img = Image.open(img_path).convert('RGB')
        
        # 應用轉換
        processed_img = process_transform(img)
        
        # 儲存處理後的圖片
        processed_img.save(output_path)
        
        return True
    except Exception as e:
        print(f"處理 {img_path} 時發生錯誤: {e}")
        return False


def process_dataset():
    """處理整個資料集"""
    print(f"\n{'='*70}")
    print("訓練資料預處理程式")
    print(f"{'='*70}")
    print(f"來源資料夾: {SOURCE_DIR}")
    print(f"輸出資料夾: {OUTPUT_DIR}")
    print(f"轉換方式: RandomHorizontalFlip + RandomRotation + ColorJitter")
    print(f"只處理訓練集 (train)，不增加圖片數量")
    print(f"{'='*70}\n")
    
    # 檢查來源資料夾
    if not os.path.isdir(SOURCE_DIR):
        print(f"錯誤: 找不到來源資料夾 {SOURCE_DIR}")
        return
    
    # 只處理 train 資料夾
    split = 'train'
    split_source_dir = os.path.join(SOURCE_DIR, split)
    split_output_dir = os.path.join(OUTPUT_DIR, split)
    
    if not os.path.isdir(split_source_dir):
        print(f"錯誤: 找不到 {split} 資料夾")
        return
    
    print(f"處理 {split.upper()} 資料集...")
    print(f"{'='*70}\n")
    
    # 取得所有類別資料夾
    class_folders = [f for f in os.listdir(split_source_dir) 
                    if os.path.isdir(os.path.join(split_source_dir, f))]
    
    total_images = 0
    total_success = 0
    
    for class_name in class_folders:
        class_source_dir = os.path.join(split_source_dir, class_name)
        class_output_dir = os.path.join(split_output_dir, class_name)
        
        # 建立輸出資料夾
        os.makedirs(class_output_dir, exist_ok=True)
        
        # 取得所有圖片檔案
        image_files = [f for f in os.listdir(class_source_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"類別: {class_name} - {len(image_files)} 張圖片")
        
        # 處理每張圖片
        success_count = 0
        for img_name in tqdm(image_files, desc=f"  處理 {class_name}"):
            img_path = os.path.join(class_source_dir, img_name)
            output_path = os.path.join(class_output_dir, img_name)
            
            if process_single_image(img_path, output_path):
                success_count += 1
        
        print(f"  完成: {success_count}/{len(image_files)} 張\n")
        
        total_images += len(image_files)
        total_success += success_count
    
    print(f"{'='*70}")
    print(f"處理完成!")
    print(f"  總共處理: {total_success}/{total_images} 張圖片")
    print(f"  輸出位置: {OUTPUT_DIR}/train/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    process_dataset()
