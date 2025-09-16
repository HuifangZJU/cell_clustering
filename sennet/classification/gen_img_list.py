import os
import random
from glob import glob

def generate_image_lists(root_dir, train_ratio=0.9, max_ratio=5, seed=42):
    random.seed(seed)

    # Paths
    snc_dir = os.path.join(root_dir, "snc")
    nonsnc_dir = os.path.join(root_dir, "non-snc")

    # Collect images
    snc_images = glob(os.path.join(snc_dir, "*"))
    nonsnc_images = glob(os.path.join(nonsnc_dir, "*"))

    # Shuffle
    random.shuffle(snc_images)
    random.shuffle(nonsnc_images)

    # Split into train/val
    snc_split = int(len(snc_images) * train_ratio)
    nonsnc_split = int(len(nonsnc_images) * train_ratio)

    snc_train, snc_val = snc_images[:snc_split], snc_images[snc_split:]
    nonsnc_train, nonsnc_val = nonsnc_images[:nonsnc_split], nonsnc_images[nonsnc_split:]

    # Generate train/val for ratios 1:1 up to max_ratio
    for ratio in range(1, max_ratio + 1):
        # Limit nonsnc count based on ratio
        n_snc_train = len(snc_train)
        n_nonsnc_train = min(len(nonsnc_train), n_snc_train * ratio)
        n_snc_val = len(snc_val)
        n_nonsnc_val = min(len(nonsnc_val), n_snc_val * ratio)

        nonsnc_train_subset = nonsnc_train[:n_nonsnc_train]
        nonsnc_val_subset = nonsnc_val[:n_nonsnc_val]

        # Build datasets
        train_set = [(os.path.relpath(img, root_dir), 1) for img in snc_train] + \
                    [(os.path.relpath(img, root_dir), 0) for img in nonsnc_train_subset]
        val_set = [(os.path.relpath(img, root_dir), 1) for img in snc_val] + \
                  [(os.path.relpath(img, root_dir), 0) for img in nonsnc_val_subset]

        # Shuffle
        random.shuffle(train_set)
        random.shuffle(val_set)

        # Save
        train_file = os.path.join(root_dir, f"train_ratio{ratio}.txt")
        val_file = os.path.join(root_dir, f"val_ratio{ratio}.txt")

        with open(train_file, "w") as f:
            for img, label in train_set:
                f.write(f"{img} {label}\n")

        with open(val_file, "w") as f:
            for img, label in val_set:
                f.write(f"{img} {label}\n")

        print(f"Saved train_ratio{ratio}.txt ({len(train_set)} samples) "
              f"and val_ratio{ratio}.txt ({len(val_set)} samples)")

# Example usage
generate_image_lists("/media/huifang/data1/sennet/codex/cell_images")
