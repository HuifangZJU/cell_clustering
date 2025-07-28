import argparse
import time
import datetime
import sys

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

from datasets import *
import torch.nn as nn
import torch
from tensorboardX import SummaryWriter
from torchvision.models import vgg16
import torch.optim as optim
from itertools import cycle


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=800, help='number of epochs of training')
parser.add_argument('--model_dir', type=str, default="cp80-patch16-image", help='name of the dataset')
parser.add_argument('--pretrained_name', type=str, default="",
                    help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=128, help='size of image height')
parser.add_argument('--img_width', type=int, default=128, help='size of image width')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=50,
                    help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=10, help='interval between model checkpoints')
args = parser.parse_args()


experiment_path = '/media/huifang/data/vizgen/classification'
image_save_path = experiment_path + '/images'
model_save_path = experiment_path + '/saved_models'
log_save_path = experiment_path + '/logs'
os.makedirs(image_save_path + '/%s' % args.model_dir, exist_ok=True)
os.makedirs(model_save_path + '/%s' % args.model_dir, exist_ok=True)
os.makedirs(log_save_path + '/%s' % args.model_dir, exist_ok=True)


# train_data_list ="/home/huifang/workspace/data/imagelists/vizgen_gene_clustered_env_image_z0_all_res0.1_train.txt"
# test_data_list ="/home/huifang/workspace/data/imagelists/vizgen_gene_clustered_env_image_z0_all_res0.1_test.txt"

train_data_list ="train_image_list.txt"
test_data_list ="test_image_list.txt"
# ------------------------------------------
#                Training preparation
# ------------------------------------------
# ------ device handling -------
# cuda = True if torch.cuda.is_available() else False
# torch.cuda.set_device(0)
# if cuda:
#     device = 'cuda'
# else:
#     device = 'cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
# ------ Configure loss -------
criterion = nn.CrossEntropyLoss()

# ------ Configure model -------
# Initialize generator and discriminator
net = vgg16(num_classes=3)
if args.epoch != 0:
    net.load_state_dict(torch.load(model_save_path +'/%s/net_%d.pth' % (args.pretrained_name, args.epoch)))
else:
    net.apply(weights_init_normal)
net.to(device)
# ------ Configure optimizer -------
# optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# ------ Configure data loaders -------
# Configure dataloaders
transform = transforms.Compose([transforms.Resize((args.img_height, args.img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5,), (0.5,))])


train_dataloader = DataLoader(ImageDataset(train_data_list, transforms=transform),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
test_dataloader = DataLoader(ImageDataset(test_data_list, transforms=transform),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)
test_samples = cycle(test_dataloader)
# Tensor type
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
def test_images():
        test_batch = next(test_samples)
        inputs, labels = test_batch[0].to(device), test_batch[1].to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return i*args.batch_size, 100 * correct / total

    # ------------------------------------------
#                Training
# ------------------------------------------
prev_time = time.time()
logger = SummaryWriter(log_save_path+'/%s' % args.model_dir)

for epoch in range(args.epoch, args.n_epochs):
    for i, batch in enumerate(train_dataloader):
        inputs, labels = batch[0].to(device),batch[1].to(device)
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # --------------
        #  Log Progress
        # --------------
        # Determine approximate time left
        batches_done = epoch * len(train_dataloader) + i
        batches_left = args.n_epochs * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r" + args.model_dir + "---[Epoch %d/%d] [Batch %d/%d] [Loss : %f]  ETA: %s" %
            (epoch, args.n_epochs,
             i, len(train_dataloader),
             loss.item(), time_left))

        # --------------tensor board--------------------------------#
        if batches_done % 50 == 0:
            cnt, acc = test_images()

            print(' Accuracy of the network on the {} validation images: {} %'.format(cnt, acc))

            info = {'loss': loss.item(),'acc': acc}
            for tag, value in info.items():
                logger.add_scalar(tag, value, batches_done)
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                logger.add_histogram(tag, value.data.cpu().numpy(), batches_done)
                # logger.add_histogram(tag+'grad', value.grad.data.cpu().numpy(),batches_done+1)

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(net.state_dict(), model_save_path+'/%s/net_%d.pth' % (args.model_dir,epoch))
# save final model
torch.save(net.state_dict(),  model_save_path+'/%s/net_%d.pth' % (args.model_dir,epoch))
logger.close()
