"""
資料擴增腳本
用於擴增訓練資料集，提升模型泛化能力
"""
import os
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import shutil

# === 配置參數 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DATA_DIR = os.path.join(BASE_DIR, "output_data")  # 原始資料夾
OUTPUT_DATA_DIR = os.path.join(BASE_DIR, "output_data_augmented")  # 擴增後資料夾
IMG_SIZE = (256, 256)  # 圖片大小
NUM_AUGMENTATIONS = 5  # 每張圖片生成的擴增版本數量

def augment_data(source_dir, output_dir, num_augmentations=5, img_size=(256, 256)):
    """
    對訓練資料進行擴增並保存到新資料夾。
    
    參數:
        source_dir (str): 原始資料夾路徑 (例如: "output_data/train")
        output_dir (str): 擴增後資料的輸出路徑 (例如: "output_data_augmented/train")
        num_augmentations (int): 每張圖片生成的擴增版本數量
        img_size (tuple): 輸出圖片大小 (height, width)
    
    回傳:
        dict: 每個類別的原始和擴增後的圖片數量統計
    """
    print(f"\n{'='*70}")
    print(f"資料擴增工具")
    print(f"{'='*70}")
    print(f"來源資料夾: {source_dir}")
    print(f"輸出資料夾: {output_dir}")
    print(f"每張圖片生成 {num_augmentations} 個擴增版本")
    print(f"圖片大小: {img_size}")
    
    # 定義多種擴增轉換組合
    augmentation_transforms = [
        # 1. 水平翻轉 + 旋轉 + 顏色調整
        T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=1.0),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]),
        # 2. 旋轉 + 顏色調整
        T.Compose([
            T.Resize(img_size),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.3, contrast=0.3),
        ]),
        # 3. 垂直翻轉 + 旋轉
        T.Compose([
            T.Resize(img_size),
            T.RandomVerticalFlip(p=1.0),
            T.RandomRotation(10),
        ]),
        # 4. 仿射變換 + 顏色調整
        T.Compose([
            T.Resize(img_size),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2),
        ]),
        # 5. 透視變換 + 旋轉
        T.Compose([
            T.Resize(img_size),
            T.RandomPerspective(distortion_scale=0.2, p=1.0),
            T.RandomRotation(10),
        ]),
        # 6. 水平翻轉 + 強烈顏色調整
        T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=1.0),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        ]),
        # 7. 旋轉 + 位移
        T.Compose([
            T.Resize(img_size),
            T.RandomRotation(25),
            T.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        ]),
        # 8. 組合變換
        T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.25, contrast=0.25),
        ]),
    ]
    
    # 檢查來源資料夾是否存在
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"找不到來源資料夾: {source_dir}")
    
    # 創建輸出資料夾
    os.makedirs(output_dir, exist_ok=True)
    
    # 獲取所有類別資料夾
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
        
        # 創建輸出類別資料夾
        os.makedirs(output_class_dir, exist_ok=True)
        
        # 獲取該類別的所有圖片
        image_files = [f for f in os.listdir(source_class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif'))]
        
        original_count = len(image_files)
        augmented_count = 0
        
        print(f"原始圖片數量: {original_count}")
        
        # 使用進度條處理每張圖片
        for img_file in tqdm(image_files, desc=f"擴增中", unit="圖片"):
            img_path = os.path.join(source_class_dir, img_file)
            
            try:
                # 讀取原始圖片
                img = Image.open(img_path).convert('RGB')
                
                # 保存原始圖片到輸出資料夾
                base_name = os.path.splitext(img_file)[0]
                ext = os.path.splitext(img_file)[1]
                
                # 保存原始圖片 (調整大小)
                original_output_path = os.path.join(output_class_dir, f"{base_name}_orig{ext}")
                img_resized = img.resize(img_size, Image.Resampling.LANCZOS)
                img_resized.save(original_output_path, quality=95)
                augmented_count += 1
                
                # 生成擴增版本
                for i in range(num_augmentations):
                    # 循環使用不同的擴增轉換
                    transform = augmentation_transforms[i % len(augmentation_transforms)]
                    
                    # 應用轉換
                    augmented_img = transform(img)
                    
                    # 保存擴增圖片
                    aug_output_path = os.path.join(output_class_dir, f"{base_name}_aug{i+1}{ext}")
                    augmented_img.save(aug_output_path, quality=95)
                    augmented_count += 1
                    
            except Exception as e:
                print(f"\n  ⚠ 警告: 處理 {img_file} 時發生錯誤: {e}")
                continue
        
        stats[class_name] = {
            'original': original_count,
            'augmented': augmented_count,
            'new_created': augmented_count - original_count
        }
        
        print(f"✓ 完成! 擴增後總圖片數: {augmented_count} (原始: {original_count}, 新增: {augmented_count - original_count})")
    
    # 印出總結
    print(f"\n{'='*70}")
    print("資料擴增完成!")
    print(f"{'='*70}\n")
    print("統計摘要:")
    print(f"{'─'*70}")
    print(f"{'類別':<20} {'原始圖片':<15} {'擴增後總數':<15} {'新增數量':<15}")
    print(f"{'─'*70}")
    
    total_original = 0
    total_augmented = 0
    total_new = 0
    
    for class_name, stat in stats.items():
        print(f"{class_name:<20} {stat['original']:<15} {stat['augmented']:<15} {stat['new_created']:<15}")
        total_original += stat['original']
        total_augmented += stat['augmented']
        total_new += stat['new_created']
    
    print(f"{'─'*70}")
    print(f"{'總計':<20} {total_original:<15} {total_augmented:<15} {total_new:<15}")
    print(f"{'─'*70}")
    print(f"\n擴增倍率: {total_augmented / total_original:.2f}x")
    print(f"{'='*70}\n")
    
    return stats


def main():
    """主程式"""
    print("\n" + "="*70)
    print("訓練資料擴增工具")
    print("="*70)
    
    # 設定路徑
    source_train_dir = os.path.join(SOURCE_DATA_DIR, "train")
    output_train_dir = os.path.join(OUTPUT_DATA_DIR, "train")
    
    # 檢查來源資料夾
    if not os.path.isdir(source_train_dir):
        print(f"\n❌ 錯誤: 找不到訓練資料夾: {source_train_dir}")
        print("請確認 output_data/train 資料夾存在")
        return
    
    try:
        # 執行資料擴增
        stats = augment_data(
            source_dir=source_train_dir,
            output_dir=output_train_dir,
            num_augmentations=NUM_AUGMENTATIONS,
            img_size=IMG_SIZE
        )
        
        # 複製驗證集和測試集 (不進行擴增)
        print("\n" + "="*70)
        print("複製驗證集和測試集...")
        print("="*70)
        
        for split in ['val', 'test']:
            source_split_dir = os.path.join(SOURCE_DATA_DIR, split)
            output_split_dir = os.path.join(OUTPUT_DATA_DIR, split)
            
            if os.path.isdir(source_split_dir):
                print(f"\n複製 {split} 資料夾...")
                if os.path.exists(output_split_dir):
                    shutil.rmtree(output_split_dir)
                shutil.copytree(source_split_dir, output_split_dir)
                
                # 統計數量
                total_images = sum([len([f for f in os.listdir(os.path.join(output_split_dir, cls)) 
                                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif'))])
                                  for cls in os.listdir(output_split_dir) 
                                  if os.path.isdir(os.path.join(output_split_dir, cls))])
                print(f"✓ {split} 複製完成 (共 {total_images} 張圖片)")
        
        print("\n" + "="*70)
        print("所有處理完成!")
        print("="*70)
        print(f"\n擴增後的完整資料集位於: {OUTPUT_DATA_DIR}")
        print(f"  - 訓練集: {output_train_dir}")
        print(f"  - 驗證集: {os.path.join(OUTPUT_DATA_DIR, 'val')}")
        print(f"  - 測試集: {os.path.join(OUTPUT_DATA_DIR, 'test')}")
        print("\n現在可以使用擴增後的資料進行訓練!")
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
