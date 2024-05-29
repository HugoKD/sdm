# coding: utf-8

import os
import random
from shutil import copyfile, move
from collections import defaultdict
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import combinations

os.getcwd()
os.listdir()

# Paths
data_dir = "data"
images_dir = "images"
annotations_dir = "annotations"
train_dir = "training"
test_dir = "test"

# Get list of all image and mask files
mask_files = [file for file in os.listdir(data_dir) if file.endswith(".png") and "_lung" in file]
image_files = [file for file in os.listdir(data_dir) if file.endswith(".png") and "_lung" not in file]

# Function to get colors from an image
def get_colors(image_path):
    with Image.open(image_path) as img:
        colors = img.getcolors()
    return [colors[i][1] for i in range(len(colors))]

heatmap = np.zeros(shape=(5,5))

color_files = defaultdict(list)
for mask_file in mask_files:
    colors = get_colors(os.path.join(data_dir, mask_file))
    for color in colors:
        color_files[color].append(mask_file)

files_color = defaultdict(list)

for mask_file in mask_files:
    colors = get_colors(os.path.join(data_dir, mask_file))
    files_color[mask_file] = colors
    for c1 in colors:
        for c2 in colors:
            heatmap[c1][c2] += 1

for i in range(10):
    print(files_color[mask_files[i]])

label_freq = {label: len(data) for label, data in color_files.items()}
print(label_freq)

df = pd.DataFrame(heatmap)
df.head()

sns.heatmap(df, annot=True, cmap="crest")

label_combinations = defaultdict(list)

for mask_file, colors in files_color.items():
    # Sort colors for consistent tuple representation
    sorted_colors = sorted(colors)
    
    label_combinations[(tuple(sorted_colors),)].append(mask_file)

# Shuffle the files for each combination
for comb in label_combinations.keys():
    random.shuffle(label_combinations[comb])

# Convert the defaultdict to a regular dictionary
label_combinations_dict = dict(label_combinations)

# Print the dictionary containing each possible set of labels and corresponding files
for labels, files in label_combinations_dict.items():
    print(f"Labels: {labels}, #Files: {len(files)}")

def image_from_mask(mask_file):
    image_file = mask_file.replace("_lung", "")
    return image_file

# Calculate the number of samples for each comb in train and test sets
train_mask_files = []
test_mask_files = []

for comb in label_combinations_dict.keys():
    total_samples = len(label_combinations_dict[comb])
    train_samples = int(0.8 * total_samples)

    train_mask_files.extend(label_combinations_dict[comb][:train_samples])
    test_mask_files.extend(label_combinations_dict[comb][train_samples:])

# Create directories for images and annotations
os.makedirs(os.path.join(images_dir, train_dir), exist_ok=True)
os.makedirs(os.path.join(images_dir, test_dir), exist_ok=True)
os.makedirs(os.path.join(annotations_dir, train_dir), exist_ok=True)
os.makedirs(os.path.join(annotations_dir, test_dir), exist_ok=True)

# Copy files to train and test directories
for mask_file in train_mask_files:
    image_file = image_from_mask(mask_file) 
    copyfile(os.path.join(data_dir, image_file), os.path.join(images_dir, train_dir, image_file))
    copyfile(os.path.join(data_dir, mask_file), os.path.join(annotations_dir, train_dir, mask_file))

# Copy files to train and test directories
for mask_file in test_mask_files:
    image_file = image_from_mask(mask_file) 
    copyfile(os.path.join(data_dir, image_file), os.path.join(images_dir, test_dir, image_file))
    copyfile(os.path.join(data_dir, mask_file), os.path.join(annotations_dir, test_dir, mask_file))

print("Train Annotations Directory: ", len([name for name in os.listdir(os.path.join(annotations_dir, train_dir))]))
print("Train Images Directory: ", len([name for name in os.listdir(os.path.join(images_dir, train_dir))]))
print("Test Annotations Directory: ", len([name for name in os.listdir(os.path.join(annotations_dir, test_dir))]))
print("Test Images Directory: ", len([name for name in os.listdir(os.path.join(images_dir, test_dir))]))
