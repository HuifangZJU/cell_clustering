import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import vgg16
from datasets import ImageDataset  # Make sure this import is correct based on your dataset handling
import numpy as np
from PIL import Image





def main():
    # Parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/media/huifang/data/vizgen/classification/saved_models/cp80-patch16-image/', help='Directory for saved models')
    parser.add_argument('--test_data_list', type=str, default='test_image_list.txt', help='Path to test image list txt file')
    parser.add_argument('--img_height', type=int, default=128, help='Image height')
    parser.add_argument('--img_width', type=int, default=128, help='Image width')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for testing')
    parser.add_argument('--n_cpu', type=int, default=8, help='Number of CPU threads for the dataloader')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the model
    model = vgg16(num_classes=3)  # Adjust num_classes according to your specific model
    model_path = f"{args.model_dir}/net_100.pth"  # Adjust the path as needed
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Set up the transformation and DataLoader
    transform = transforms.Compose([
        transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load test data
    test_dataset = ImageDataset(args.test_data_list, transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu)

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print the accuracy
    print(f'Accuracy of the network on the {total} test images: {100 * correct / total:.2f} %')

if __name__ == "__main__":
    main()
