import os, random

def generate_image_lists(root_dir, train_ratio=0.9, seed=42):
    random.seed(seed)

    # collect all .png files
    images = [f for f in os.listdir(root_dir+"all") if f.endswith(".png")]

    # shuffle
    random.shuffle(images)

    # split train/test
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    test_images = images[split_idx:]

    def parse_label(filename):
        # filename format: sampleid_regionid_cellid_score.png
        score_str = filename.rsplit("_", 1)[-1].replace(".png", "")
        return float(score_str)

    def write_list(filelist, filename):
        with open(os.path.join(root_dir, filename), "w") as f:
            for img in filelist:
                label = parse_label(img)
                f.write(f"{img} {label}\n")

    # save
    write_list(train_images, root_dir + "regression_train.txt")
    write_list(test_images, root_dir + "regression_val.txt")

    print(f"Saved train.txt ({len(train_images)} images) and test.txt ({len(test_images)} images) in {root_dir}")

# Example usage
generate_image_lists("/media/huifang/data/sennet/codex/cell_images/")
