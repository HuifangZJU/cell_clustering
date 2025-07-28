"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np

from utils.config import create_config
from utils.common_config import get_criterion, get_model, get_train_dataset,\
                                get_val_dataset, get_train_dataloader,\
                                get_val_dataloader, get_train_transformations,\
                                get_val_transformations, get_optimizer,\
                                adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate
from utils.memory import MemoryBank
from utils.train_utils import simclr_train
from utils.utils import fill_memory_bank
from termcolor import colored
import time
from tensorboardX import SummaryWriter
import warnings

# Use simplefilter to ignore a specific warning category
warnings.filterwarnings("ignore")

# Parser
parser = argparse.ArgumentParser(description='SimCLR')
parser.add_argument('--config_env', default='configs/env.yml',
                    help='Config file for the environment')
parser.add_argument('--config_exp',default='configs/simclr_vizgen.yml',
                    help='Config file for the experiment')
parser.add_argument('--ds_rate',default=1,type=int,
                    help='data downsample rate')
parser.add_argument('--batch_size',default=8,type=int,
                    help='data batchsize')
args = parser.parse_args()

def main():

    # Retrieve config file
    p = create_config(args.config_env, args.config_exp,args.ds_rate,args.batch_size)
    print(colored(p, 'red'))
    start_time = time.time()
    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    # print(model)
    model = model.cuda()
   
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True
    
    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    # print('Train transforms:', train_transforms)
    train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True)
    train_dataloader = get_train_dataloader(p, train_dataset)
    print('Dataset contains {} train samples'.format(len(train_dataset)))


    # # Memory Bank
    # print(colored('Build MemoryBank', 'blue'))
    # val_transformer = get_val_transformations(p)
    # base_dataset = get_val_dataset(p,val_transformer) # Dataset w/o augs for knn eval
    # base_dataloader = get_val_dataloader(p, base_dataset)
    # memory_bank_base = MemoryBank(len(base_dataset),
    #                             p['model_kwargs']['features_dim'],
    #                             p['num_classes'], p['criterion_kwargs']['temperature'])
    #
    # memory_bank_base.cuda()

    # Criterion
    # print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    # print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    # print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    # print(optimizer)


    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()

    # Training
    print(colored('Starting main loop', 'blue'))

    logger = SummaryWriter(p['log_path'])
    logger_base = 50
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        
        # Train
        print('Train ...')
        # simclr_train(train_dataloader, model, criterion, optimizer, epoch,logger)
        model.train()
        for i, batch in enumerate(train_dataloader):
            print(i)
            images = batch['image']

            images_augmented = batch['image_augmented']
            b, c, h, w = images.size()
            input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)

            input_ = input_.view(-1, c, h, w)
            input_ = input_.cuda(non_blocking=True)
            output = model(input_).view(b, 2, -1)
            loss = criterion(output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % logger_base == 0:
                info = {'loss': loss.item()}
                iter = epoch*len(train_dataloader)+i
                for tag, value in info.items():
                    logger.add_scalar(tag, value, iter)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.add_histogram(tag, value.data.cpu().numpy(), iter)
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1}, p['pretext_checkpoint'])

        if epoch % 100 == 0:
            torch.save(model.state_dict(), p['pretext_dir']+'/model-'+str(epoch)+'-'+str(epoch*len(train_dataloader))+'.pth.tar')
    logger.close()

    # Save final model


    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    # print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    # fill_memory_bank(base_dataloader, model, memory_bank_base)
    # memory_bank_base.cpu()
    #
    # suffix = 'ds' + str(p['ds_rate']) + '-bs' + str(p['batch_size'])
    # torch.save({'feature':memory_bank_base.features,'target':memory_bank_base.targets,
    #             'position_x':memory_bank_base.position_x,'position_y':memory_bank_base.position_y},'simclr'+suffix+'.pt')
    #
    #
    # topk = 5
    # print('Mine the nearest neighbors (Top-%d)' %(topk))
    # indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    # print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    # end_time = time.time()
    #
    # # Calculate the elapsed time
    # elapsed_time = end_time - start_time
    # print(f"Total running time: {elapsed_time:.2f} seconds")
    # np.save(p['topk_neighbors_train_path'], indices)
 
if __name__ == '__main__':
    main()
