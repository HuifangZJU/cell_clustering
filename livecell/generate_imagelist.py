import os


def generate_image_list(parent_folder):
    image_list = []
    label = 1

    if not os.path.exists(parent_folder):
        raise ValueError(f"Parent folder '{parent_folder}' does not exist.")

    subfolders = [subfolder for subfolder in os.listdir(parent_folder) if
                  os.path.isdir(os.path.join(parent_folder, subfolder))]
    for subfolder in subfolders:
        if subfolder not in ['bv2', 'huh7','shsy5y', 'skbr3']:
            continue
        image_folder = os.path.join(parent_folder, subfolder)

        images_in_folder = os.listdir(image_folder)
        for image_filename in images_in_folder:
            image_path = os.path.join(image_folder, image_filename)
            image_list.append((image_path, label))

        label += 1

    return image_list


def save_image_list_to_txt(image_list, txt_file):
    with open(txt_file, 'w') as f:
        for image_path, label in image_list:
            f.write(f"{image_path} {label}\n")


# Example usage with the parent folder that contains the four image folders
# train: ['a172', 'bt474', 'bv2', 'huh7', 'mcf7', 'shsy5y', 'skbr3', 'skov3']
parent_folder = '/media/huifang/data/livecell/LIVECell_dataset_2021/single_cell_images/test'
image_list = generate_image_list(parent_folder)

# Save the image list to a txt file
txt_file = '/media/huifang/data/livecell/LIVECell_dataset_2021/single_cell_images/test-4class.txt'
save_image_list_to_txt(image_list, txt_file)