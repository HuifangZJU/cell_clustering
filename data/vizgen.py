"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
from PIL import Image
from torch.utils.data import Dataset
from utils.mypath import MyPath

class vizgen(Dataset):

    def __init__(self, root=MyPath.db_root_dir('vizgen'), split='train',transform=None,ds_rate=1):
        super(vizgen, self).__init__()
        self.root = root
        self.transform = transform
        if split == 'train':
            # f = open(root+'vizgen_breast_local_image_z0_all_res0.1_ds_train_with_position.txt')
            # f = open(root + 'vizgen_breast_local_image_z0_all_res0.1_ds_all_with_size.txt', 'r')
            # f = open(root+ 'vizgen_patches/cellpose/cellpose_colon_patch16.txt')
            f = open(root+'vizgen_colon_z0_ds1_res0.1_cellpose_3class.txt')
        elif split == 'test':

            # f = open(root + 'vizgen_breast_local_image_z0_all_res0.1_ds_train_with_position.txt')
            f = open(root+ 'vizgen_patches/cellpose/cellpose_colon_patch16.txt')
        self.data = f.readlines()
        self.data = self.data[::ds_rate]
        self.classes=['cancer','stromal','immune']

    def __getitem__(self, index):
        # make consistent with all other datasets
        # return a PIL Image
        img_path = self.data[index % len(self.data)].strip()
        img_path = img_path.split(' ')

        img = Image.open(img_path[0])
        img_size = img.size
        if img_size[0]<96:
            img = img.resize((96,96))
        if self.transform is not None:
            img = self.transform(img)

        target = int(img_path[1])
        # position_y = float(img_path[2])
        # position_x = float(img_path[3])
        size = float(img_path[2])/10000
        class_name = self.classes[target]
        out = {'image': img, 'target': target,'position_x':size,'position_y':size,'meta': {'im_size': img_size, 'index': index, 'class_name': class_name}}
        return out

    def __len__(self):
        return len(self.data)