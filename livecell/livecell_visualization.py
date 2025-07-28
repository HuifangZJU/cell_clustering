import json
import os

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw

def show_image_with_annotations(image_path, annotations, category_id_to_color):
    # Load the image
    image = Image.open(image_path)

    # Create a figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image,cmap='gray')

    # Plot segmentation masks with different colors
    for ann in annotations:
        category_id = ann['category_id']
        color = category_id_to_color.get(category_id,
                                         'red')  # Default to red color if category ID not found in the mapping

        # Get the segmentation data (list of polygons) for the annotation
        segmentations = ann['segmentation']

        for poly in segmentations:
            # Reshape the flat list of coordinates into an array with two columns for x and y
            xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]

            # Create a Polygon patch for each segmentation and add it to the plot
            polygon = patches.Polygon(xy, closed=True, edgecolor=color, facecolor='none')
            ax.add_patch(polygon)

    # Show the image with segmentation masks
    plt.show()


def crop_and_save_images(image_path, image_annotations, output_folder):
    os.makedirs(output_folder,exist_ok=True)
    image = Image.open(image_path)
    # Iterate through each annotation and crop the image
    # Calculate the average gray value of the whole image
    avg_gray_value = int(np.mean(np.array(image)))

    # Iterate through each annotation and crop the image
    for ann in image_annotations:
        # Get the bounding box coordinates [x, y, width, height]
        bbox = ann['bbox']
        x, y, width, height = map(int, bbox)

        # Crop the image using the bounding box coordinates
        cropped_image = image.crop((x, y, x + width, y + height))

        # Calculate the center of the cropped image
        center_x, center_y = width // 2, height // 2

        # Calculate the size of the output image [128, 128]
        # output_size = (96, 96)
        #
        # if width < output_size[0] or height < output_size[1]:
        #     # Extract the COCO segmentation polygon points
        #     segmentation = ann['segmentation']
        #     polygon_points = []
        #     for poly in segmentation:
        #         polygon_points.extend(poly)
        #
        #     # Create a new image with [128, 128] size and average gray value
        #     background_image = Image.new('L', output_size, color=avg_gray_value)
        #
        #     # Convert the polygon points into pairs of (x, y) coordinates
        #     polygon_points = [(polygon_points[i], polygon_points[i + 1]) for i in range(0, len(polygon_points), 2)]
        #
        #     # Draw the COCO segmentation polygon on the new image
        #     draw = ImageDraw.Draw(background_image)
        #     draw.polygon(polygon_points, fill=255)
        #
        #     # Paste the cropped image onto the new image using the center coordinates
        #     paste_position = (output_size[0] // 2 - center_x, output_size[1] // 2 - center_y)
        #     background_image.paste(cropped_image, paste_position)
        #
        #     # Resize the new image to [128, 128]
        #     final_image = background_image.resize(output_size)
        # else:
        #     # Resize the cropped image to [128, 128]
        #     final_image = cropped_image.resize(output_size)


        # Save the cropped and resized image to a file with a unique name based on the image ID and annotation ID
        cropped_image_filename = f"{image_id}_annotation_{ann['id']}.jpg"
        cropped_image_path = f"{output_folder}/{cropped_image_filename}"
        # final_image.save(cropped_image_path)
        cropped_image.save(cropped_image_path)

#celltypes=['bv2','huh7','shsy5y','skov3']
#imagetypss=['BV2','Huh7','SHSY5Y','SKOV3']
celltypes=['a172','bt474','mcf7','skbr3','bv2','huh7','shsy5y','skov3']
imagetypss=['A172','BT474','MCF7','SkBr3','BV2','Huh7','SHSY5Y','SKOV3']
# Load COCO JSON file
for celltype,imgtype in zip(celltypes,imagetypss):
    print(celltype)
    json_file_path = f'/media/huifang/data/livecell/LIVECell_dataset_2021/annotations/LIVECell_single_cells/{celltype}/train.json'
    image_path = f"/media/huifang/data/livecell/LIVECell_dataset_2021/images/livecell_train_val_images/{imgtype}/"
    with open(json_file_path, 'r') as f:
        coco_data = json.load(f)

    # Access the images and annotations
    images = coco_data['images']
    print(len(images))
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    # Create a mapping of category IDs to colors (customize this based on your requirement)
    category_id_to_color = {category['id']: category.get('color', 'red') for category in categories}

    # Iterate through each image and display it with segmentation masks and different colors for each label
    # for i in range(0,len(images),10):
    #     image_info = images[i]
    for image_info in images:
        image_id = image_info['id']
        image_filename = image_info['file_name']

        # Filter annotations for the current image
        image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]

        # Show the image with segmentation masks and different colors for each label
        show_image_with_annotations(image_path+image_filename, image_annotations, category_id_to_color)
        # crop_and_save_images(image_path+image_filename,image_annotations,f'/media/huifang/data/livecell/LIVECell_dataset_2021/single_cell_bbox/{celltype}')

