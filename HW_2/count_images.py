"""
計算資料夾中的圖片數量
"""
import os

def count_images_in_folder(folder_path):
    """計算資料夾中各類別的圖片數量"""
    if not os.path.isdir(folder_path):
        print(f"❌ 錯誤: 找不到資料夾 {folder_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"統計資料夾: {folder_path}")
    print(f"{'='*70}\n")
    
    # 獲取所有類別資料夾
    classes = [d for d in os.listdir(folder_path) 
               if os.path.isdir(os.path.join(folder_path, d))]
    
    if len(classes) == 0:
        print("找不到任何類別資料夾")
        return
    
    print(f"找到 {len(classes)} 個類別\n")
    print(f"{'─'*70}")
    print(f"{'類別':<30} {'圖片數量':>10}")
    print(f"{'─'*70}")
    
    total_images = 0
    class_counts = {}
    
    for class_name in sorted(classes):
        class_dir = os.path.join(folder_path, class_name)
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif'))]
        count = len(images)
        class_counts[class_name] = count
        total_images += count
        print(f"{class_name:<30} {count:>10}")
    
    print(f"{'─'*70}")
    print(f"{'總計':<30} {total_images:>10}")
    print(f"{'─'*70}\n")
    
    # 檢查是否平衡
    if len(set(class_counts.values())) == 1:
        print(f"✓ 所有類別的圖片數量都相同 ({list(class_counts.values())[0]} 張)")
    else:
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        print(f"⚠ 類別數量不一致")
        print(f"  最少: {min_count} 張")
        print(f"  最多: {max_count} 張")
        print(f"  差異: {max_count - min_count} 張")
    
    print(f"\n{'='*70}\n")
    
    return class_counts, total_images


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 統計 output_data_augmented/train
    train_folder = os.path.join(BASE_DIR, "output_data_augmented", "train")
    count_images_in_folder(train_folder)
