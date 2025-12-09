"""
資料平衡工具
將訓練集中各類別的圖片數量平衡到相同數量
"""
import os
import shutil
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import random

# === 配置參數 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_DIR = os.path.join(BASE_DIR, "output_data", "train")
OUTPUT_DATA_DIR = os.path.join(BASE_DIR, "output_data_balanced", "train")
IMG_SIZE = (256, 256)

# 平衡策略
# 'max': 將所有類別擴增到最多的類別數量
# 'min': 將所有類別減少到最少的類別數量
# 'custom': 自訂目標數量
BALANCE_STRATEGY = 'max'
# CUSTOM_TARGET_COUNT = 500  # 當 BALANCE_STRATEGY='custom' 時使用


def count_images_per_class(data_dir):
    """統計每個類別的圖片數量"""
    classes = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    
    class_counts = {}
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif'))]
        class_counts[class_name] = len(images)
    
    return class_counts


def augment_image(img, img_size=(256, 256)):
    """對單張圖片進行隨機擴增"""
    augmentation_transforms = [
        T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=1.0),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]),
        T.Compose([
            T.Resize(img_size),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.3, contrast=0.3),
        ]),
        T.Compose([
            T.Resize(img_size),
            T.RandomVerticalFlip(p=1.0),
            T.RandomRotation(10),
        ]),
        T.Compose([
            T.Resize(img_size),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2),
        ]),
        T.Compose([
            T.Resize(img_size),
            T.RandomPerspective(distortion_scale=0.2, p=1.0),
            T.RandomRotation(10),
        ]),
        T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=1.0),
            T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            T.RandomRotation(12),
        ]),
    ]
    
    # 隨機選擇一個轉換
    transform = random.choice(augmentation_transforms)
    return transform(img)


def balance_dataset(source_dir, output_dir, target_count, img_size=(256, 256)):
    """
    平衡資料集,使每個類別的圖片數量相同
    
    參數:
        source_dir: 原始訓練資料夾
        output_dir: 輸出資料夾
        target_count: 目標圖片數量
        img_size: 圖片大小
    """
    print(f"\n{'='*70}")
    print("資料平衡工具")
    print(f"{'='*70}")
    print(f"來源資料夾: {source_dir}")
    print(f"輸出資料夾: {output_dir}")
    print(f"目標圖片數量: {target_count}")
    
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"找不到來源資料夾: {source_dir}")
    
    # 創建輸出資料夾
    os.makedirs(output_dir, exist_ok=True)
    
    # 獲取所有類別
    classes = [d for d in os.listdir(source_dir) 
               if os.path.isdir(os.path.join(source_dir, d))]
    
    if len(classes) == 0:
        raise RuntimeError(f"在 {source_dir} 中找不到任何類別資料夾")
    
    print(f"\n找到 {len(classes)} 個類別: {classes}\n")
    
    stats = {}
    
    # 處理每個類別
    for class_name in classes:
        print(f"{'─'*70}")
        print(f"處理類別: {class_name}")
        print(f"{'─'*70}")
        
        source_class_dir = os.path.join(source_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # 獲取該類別的所有圖片
        image_files = [f for f in os.listdir(source_class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif'))]
        
        original_count = len(image_files)
        print(f"原始圖片數量: {original_count}")
        print(f"目標圖片數量: {target_count}")
        
        if original_count == 0:
            print(f"⚠ 警告: {class_name} 類別沒有圖片,跳過")
            continue
        
        output_count = 0
        
        if original_count >= target_count:
            # 數量過多,隨機抽樣
            print(f"策略: 隨機抽樣 (減少 {original_count - target_count} 張)")
            sampled_files = random.sample(image_files, target_count)
            
            for img_file in tqdm(sampled_files, desc="複製中"):
                src_path = os.path.join(source_class_dir, img_file)
                dst_path = os.path.join(output_class_dir, img_file)
                
                try:
                    img = Image.open(src_path).convert('RGB')
                    img_resized = img.resize(img_size, Image.Resampling.LANCZOS)
                    img_resized.save(dst_path, quality=95)
                    output_count += 1
                except Exception as e:
                    print(f"\n⚠ 警告: 處理 {img_file} 時發生錯誤: {e}")
        
        else:
            # 數量不足,需要擴增
            print(f"策略: 資料擴增 (增加 {target_count - original_count} 張)")
            
            # 先複製所有原始圖片
            print("步驟 1/2: 複製原始圖片...")
            for img_file in tqdm(image_files, desc="複製中"):
                src_path = os.path.join(source_class_dir, img_file)
                base_name = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1]
                dst_path = os.path.join(output_class_dir, f"{base_name}_orig{ext}")
                
                try:
                    img = Image.open(src_path).convert('RGB')
                    img_resized = img.resize(img_size, Image.Resampling.LANCZOS)
                    img_resized.save(dst_path, quality=95)
                    output_count += 1
                except Exception as e:
                    print(f"\n⚠ 警告: 處理 {img_file} 時發生錯誤: {e}")
            
            # 擴增圖片直到達到目標數量
            needed = target_count - output_count
            print(f"步驟 2/2: 生成 {needed} 張擴增圖片...")
            
            aug_count = 0
            with tqdm(total=needed, desc="擴增中") as pbar:
                while output_count < target_count:
                    # 隨機選擇一張原始圖片進行擴增
                    img_file = random.choice(image_files)
                    src_path = os.path.join(source_class_dir, img_file)
                    
                    try:
                        img = Image.open(src_path).convert('RGB')
                        augmented_img = augment_image(img, img_size)
                        
                        base_name = os.path.splitext(img_file)[0]
                        ext = os.path.splitext(img_file)[1]
                        aug_path = os.path.join(output_class_dir, f"{base_name}_aug{aug_count}{ext}")
                        
                        augmented_img.save(aug_path, quality=95)
                        output_count += 1
                        aug_count += 1
                        pbar.update(1)
                        
                    except Exception as e:
                        print(f"\n⚠ 警告: 擴增 {img_file} 時發生錯誤: {e}")
                        continue
        
        stats[class_name] = {
            'original': original_count,
            'final': output_count,
            'change': output_count - original_count
        }
        
        print(f"✓ 完成! 最終圖片數: {output_count}")
    
    # 印出總結
    print(f"\n{'='*70}")
    print("資料平衡完成!")
    print(f"{'='*70}\n")
    print("統計摘要:")
    print(f"{'─'*70}")
    print(f"{'類別':<20} {'原始數量':<15} {'最終數量':<15} {'變化':<15}")
    print(f"{'─'*70}")
    
    for class_name, stat in stats.items():
        change_str = f"+{stat['change']}" if stat['change'] > 0 else str(stat['change'])
        print(f"{class_name:<20} {stat['original']:<15} {stat['final']:<15} {change_str:<15}")
    
    print(f"{'─'*70}")
    total_final = sum(s['final'] for s in stats.values())
    print(f"{'總計':<20} {'':<15} {total_final:<15}")
    print(f"{'─'*70}\n")
    
    # 驗證所有類別數量是否相同
    final_counts = [s['final'] for s in stats.values()]
    if len(set(final_counts)) == 1:
        print(f"✓ 驗證通過: 所有類別的圖片數量都是 {final_counts[0]}")
    else:
        print(f"⚠ 警告: 類別數量不一致: {final_counts}")
    
    return stats


def main():
    """主程式"""
    print("\n" + "="*70)
    print("訓練資料平衡工具")
    print("="*70)
    
    # 檢查來源資料夾
    if not os.path.isdir(SOURCE_DATA_DIR):
        print(f"\n❌ 錯誤: 找不到訓練資料夾: {SOURCE_DATA_DIR}")
        print("請確認 output_data/train 資料夾存在")
        return
    
    # 統計當前類別數量
    print("\n當前資料分布:")
    print("─"*70)
    class_counts = count_images_per_class(SOURCE_DATA_DIR)
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name:<30} {count:>5} 張")
    print("─"*70)
    
    if len(class_counts) == 0:
        print("\n❌ 錯誤: 找不到任何類別資料")
        return
    
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    avg_count = sum(class_counts.values()) // len(class_counts)
    
    print(f"\n最少: {min_count} 張")
    print(f"最多: {max_count} 張")
    print(f"平均: {avg_count} 張")
    
    # 決定目標數量
    if BALANCE_STRATEGY == 'max':
        target_count = max_count
        print(f"\n策略: 將所有類別擴增到 {target_count} 張 (最大值)")
    elif BALANCE_STRATEGY == 'min':
        target_count = min_count
        print(f"\n策略: 將所有類別減少到 {target_count} 張 (最小值)")
    # else:  # custom
    #     target_count = CUSTOM_TARGET_COUNT
    #     print(f"\n策略: 將所有類別調整到 {target_count} 張 (自訂)")
    
    # 執行平衡
    try:
        stats = balance_dataset(
            source_dir=SOURCE_DATA_DIR,
            output_dir=OUTPUT_DATA_DIR,
            target_count=target_count,
            img_size=IMG_SIZE
        )
        
        # 複製驗證集和測試集
        print("\n" + "="*70)
        print("複製驗證集和測試集...")
        print("="*70)
        
        base_data_dir = os.path.dirname(SOURCE_DATA_DIR)
        output_base_dir = os.path.dirname(OUTPUT_DATA_DIR)
        
        for split in ['val', 'test']:
            source_split_dir = os.path.join(base_data_dir, split)
            output_split_dir = os.path.join(output_base_dir, split)
            
            if os.path.isdir(source_split_dir):
                print(f"\n複製 {split} 資料夾...")
                if os.path.exists(output_split_dir):
                    shutil.rmtree(output_split_dir)
                shutil.copytree(source_split_dir, output_split_dir)
                
                # 統計數量
                split_counts = count_images_per_class(output_split_dir)
                total_images = sum(split_counts.values())
                print(f"✓ {split} 複製完成 (共 {total_images} 張圖片)")
                for cls, cnt in sorted(split_counts.items()):
                    print(f"    {cls}: {cnt} 張")
        
        print("\n" + "="*70)
        print("所有處理完成!")
        print("="*70)
        print(f"\n平衡後的完整資料集位於: {output_base_dir}")
        print(f"  - 訓練集 (已平衡): {OUTPUT_DATA_DIR}")
        print(f"  - 驗證集: {os.path.join(output_base_dir, 'val')}")
        print(f"  - 測試集: {os.path.join(output_base_dir, 'test')}")
        print("\n現在可以使用平衡後的資料進行訓練!")
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 設定隨機種子以確保可重現性
    random.seed(42)
    main()
