import cv2
import os
import shutil

from tqdm import tqdm

source = "/home/shravan/deep-learning/data/PPP_orig_cleaned"
dest = "/home/shravan/deep-learning/data/PPP_orig_cleaned_filtered"

dir = os.listdir(source)

if not os.path.exists(dest):
    for d in dir:
        os.makedirs(os.path.join(dest, d))

    count = 0
    for img in tqdm(os.listdir(os.path.join(source, "images"))):
        im = cv2.imread(os.path.join(source, "images", img), cv2.IMREAD_UNCHANGED)
        h, w = im.shape[:-1]
        if (h >= 1000 and w >= 2000) or (h >= 2000 and w >= 1000):
            for d in dir:
                f = img.replace("jpg", "png") if not os.path.exists(os.path.join(source, d, img)) else img
                shutil.copy(os.path.join(source, d, f), os.path.join(dest, d, f))
        else:
            count += 1
    print(count, "were not copied.")
else:
    print("destination directory already exists, not copying ...")
    count = 0
    idx_rm = [1,55,68,96,140,152]
    for img in tqdm(os.listdir(os.path.join(dest, "images"))):
        if any([img.endswith('0'+str(id)+'.jpg') for id in idx_rm]):
            print("removing", img)
            for d in dir:
                f = img.replace("jpg", "png") if not os.path.exists(os.path.join(dest, d, img)) else img
                os.remove(os.path.join(dest, d, f))
            count += 1
    print(count,"/",len(idx_rm), "img were removed")