import numpy as np

def load_image_paths(file_path):
    """ Load image paths from a text file. """
    with open(file_path, 'r') as file:
        image_paths = file.readlines()
    return image_paths

def filter_images_by_ids(image_paths, ids):
    """ Filter images based on provided IDs. """
    # Filter images; note that IDs should match the line numbers minus one (0-indexed)
    return [image_paths[i] for i in ids]

def save_image_paths(file_path, image_paths):
    """ Save image paths to a text file. """
    with open(file_path, 'w') as file:
        for path in image_paths:
            file.write(path)

def main():
    # Paths to your files
    original_image_list_path = '/home/huifang/workspace/data/imagelists/vizgen_patches/cellpose/cellpose_colon_patch16.txt'
    train_ids_path = '/media/huifang/data/experiment/tokencut/cp80-patch16/train_id.npy'
    test_ids_path = '/media/huifang/data/experiment/tokencut/cp80-patch16/test_id.npy'

    # Output paths
    output_train_image_list_path = 'train_image_list.txt'
    output_test_image_list_path = 'test_image_list.txt'

    # Load data
    image_paths = load_image_paths(original_image_list_path)
    train_ids = np.load(train_ids_path)  # Assuming IDs are saved as numpy arrays
    test_ids = np.load(test_ids_path)

    # Ensure IDs are zero-indexed for Python list
    train_ids = [int(id) for id in train_ids]
    test_ids = [int(id) for id in test_ids]

    # Filter images
    train_image_paths = filter_images_by_ids(image_paths, train_ids)
    test_image_paths = filter_images_by_ids(image_paths, test_ids)

    # Save filtered image paths
    save_image_paths(output_train_image_list_path, train_image_paths)
    save_image_paths(output_test_image_list_path, test_image_paths)

    print("Image lists have been split and saved successfully.")

if __name__ == '__main__':
    main()
