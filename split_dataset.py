import os
import random
import shutil
image_folder = 'combined'
label_folder = 'combined_output'
train_folder = 'train'
val_folder = 'val'
test_folder = 'test'
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
image_files = os.listdir(image_folder)
label_files = os.listdir(label_folder)
random.shuffle(image_files)
train_split = 0.7
val_split = 0.15
test_split = 0.15
total_images = len(image_files)
train_end = int(total_images * train_split)
val_end = train_end + int(total_images * val_split)
train_images = image_files[:train_end]
val_images = image_files[train_end:val_end]
test_images = image_files[val_end:]

def move_files(images, source_folder, dest_folder):
    for image in images:
        label_file = image[:-4] + ".json"  
        if label_file in label_files:
            shutil.move(os.path.join(image_folder, image), os.path.join(dest_folder, image))
            shutil.move(os.path.join(label_folder, label_file), os.path.join(dest_folder, label_file))

move_files(train_images, image_folder, train_folder)
move_files(val_images, image_folder, val_folder)
move_files(test_images, image_folder, test_folder)

print("Dataset split successfully.")
