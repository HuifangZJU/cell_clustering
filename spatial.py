"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch

from termcolor import colored
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_optimizer, get_model, get_criterion,\
                                adjust_learning_rate

from tensorboardX import SummaryWriter
from utils.memory import MemoryBank
from utils.utils import fill_memory_bank_with_pretrained_feature


def get_nearest_neighbor_acc(spatial_model,positions,features,image_memory_bank):
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    spatial_model._modules['encoder']._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
    # Forward pass in the model
    attentions = spatial_model._modules['encoder'].get_last_selfattention(positions.to(torch.float),features.to(torch.float))
    last_y, last_x = spatial_model._modules['encoder'].get_last_y(positions.to(torch.float),features.to(torch.float))

    # Dimensions
    nb_im = attentions.shape[0]  # Batch size
    nh = attentions.shape[1]  # Number of heads
    nb_tokens = attentions.shape[2]  # Number of tokens
    qkv = (
        feat_out["qkv"]
        .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)
    v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)

    accuracies=[]
    for feats in [q,k,v,last_y, last_x]:
        feats = feats.squeeze().detach()
        image_memory_bank.features = feats
        topk = 5
        print('Mine the nearest neighbors (Top-%d)' % (topk))
        indices, acc = image_memory_bank.mine_nearest_neighbors(topk)
        accuracies.append(acc)
        print('Accuracy of top-%d nearest neighbors on train set is %.2f' % (topk, 100 * acc))
    return accuracies



FLAGS = argparse.ArgumentParser(description='spaformer Loss')
FLAGS.add_argument('--config_env', default='configs/env.yml', help='Location of path config file')
FLAGS.add_argument('--config_exp', default='configs/spatial_vizgen.yml', help='Location of experiments config file')

def main():
    args = FLAGS.parse_args()
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))


    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    train_dataset = get_train_dataset(p, train_transformations,
                                      split='train', to_neighbors_dataset=True)
    train_dataloader = get_train_dataloader(p, train_dataset)
    print('Train transforms:', train_transformations)
    print('Train samples %d.' % len(train_dataset))

    
    # Model
    print(colored('Get model', 'blue'))
    model = get_model(p, p['pretext_model'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['update_spatial_head_only'])

    
    # Warning
    if p['update_spatial_head_only']:
        print(colored('WARNING: Spaformer will only update the cluster head', 'red'))

    # Loss function
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p) 
    criterion.cuda()

    # Main loop
    print(colored('Starting main loop', 'blue'))
    experiment_path = '/media/huifang/data/experiment/tokencut'
    model_save_path = experiment_path + '/%s/saved_models' % p['model_dir']
    log_save_path = experiment_path + '/%s/logs' % p['model_dir']
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)
    logger = SummaryWriter(log_save_path)

    image_memory_bank = MemoryBank(len(train_dataset), 128,3,0.1)
    simclr_feature = torch.load('simclr.pt')
    fill_memory_bank_with_pretrained_feature(image_memory_bank, simclr_feature)
    for epoch in range(0, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        for i, batch in enumerate(train_dataloader):
            images = batch['image']
            position_x = batch['position_x']
            position_y = batch['position_y']
            positions = torch.cat((position_x, position_y), dim=1)

            print(positions.shape)
            print(images.shape)
            test = input()

            features,loss = model(positions.to(torch.float), images.to(torch.float))

            print(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            # --------------tensor board--------------------------------#
            if epoch % 100 == 0:
                accuracies = get_nearest_neighbor_acc(model,positions,features,image_memory_bank)
                info = {'loss': loss.item(), 'q': accuracies[0], 'k': accuracies[1], 'v': accuracies[2],
                        'last_y': accuracies[3], 'last_x': accuracies[4]}
                for tag, value in info.items():
                    logger.add_scalar(tag, value, epoch)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.add_histogram(tag, value.data.cpu().numpy(), epoch)

        if epoch % 5000 == 0:
            torch.save(model.state_dict(), model_save_path + '/spaformer_%d.pth' % (epoch))
    logger.close()


    
if __name__ == "__main__":
    main()
