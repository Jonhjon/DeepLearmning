import os
import csv

BASE = r"c:\Users\H514 #4856\Desktop\deep_learning_114206103\HW2\output_data"
SPLITS = ['train', 'val', 'test']
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif'}

# collect class names from train if available, else union of subdirs
classes = set()
for s in SPLITS:
    p = os.path.join(BASE, s)
    if os.path.isdir(p):
        for d in os.listdir(p):
            if os.path.isdir(os.path.join(p, d)):
                classes.add(d)

classes = sorted(classes)
if not classes:
    print('找不到任何 class 子資料夾，請確認路徑：', BASE)
    raise SystemExit(1)

results = {}
for cls in classes:
    row = {"train": 0, "val": 0, "test": 0}
    total = 0
    for s in SPLITS:
        p = os.path.join(BASE, s, cls)
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
            # count only image-like extensions
            n = sum(1 for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
            row[s] = n
            total += n
        else:
            row[s] = 0
    row['total'] = total
    results[cls] = row

# totals
totals = {"train": 0, "val": 0, "test": 0, 'total': 0}
for cls, r in results.items():
    for k in totals:
        totals[k] += r.get(k, 0)

# write CSV

# print summary
print('Corn_Leaves_Split 影像數量統計:')
for cls in classes:
    r = results[cls]
    print(f"  {cls}: train={r['train']}, val={r['val']}, test={r['test']}, total={r['total']}")
print('總計: train={train}, val={val}, test={test}, total={total}'.format(**totals))
