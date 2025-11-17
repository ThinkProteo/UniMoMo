#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
import yaml
import torch
from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
from utils.config_utils import overwrite_values
from utils import register as R
########### Import your packages below ##########
from trainer import create_trainer
from data import create_dataset, create_dataloader
from utils.nn_utils import count_parameters
import models

def parse():
    parser = argparse.ArgumentParser(description='training')
    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    # config
    parser.add_argument('--config', type=str, required=True, help='Path to the yaml configure')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    
    # ADDED: Resume from checkpoint
    parser.add_argument('--resume_from_checkpoint', type=str, default='', 
                        help='Path to checkpoint to resume training from')
    
    return parser.parse_known_args()

def main(args, opt_args):
    # load config
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)
    
    # ADDED: Store resume checkpoint path in config
    resume_checkpoint = args.resume_from_checkpoint or config.get('resume_from_checkpoint', '')
    
    ########## define your model #########
    model = R.construct(config['model'])
    
    # Load weights only (for transfer learning/pretrained models)
    if len(config.get('load_ckpt', '')):
        checkpoint = torch.load(config['load_ckpt'], map_location='cpu')
        if hasattr(checkpoint, 'state_dict'):
            model.load_state_dict(checkpoint.state_dict())
        else:
            model.load_state_dict(checkpoint)
        print_log(f'Loaded weights from {config["load_ckpt"]}')
    
    ########### load your train / valid set ###########
    train_set, valid_set, _ = create_dataset(config['dataset'])
    
    ########## define your trainer/trainconfig #########
    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=int(os.environ['WORLD_SIZE']))
    else:
        args.local_rank = -1
    
    if args.local_rank <= 0:
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
    
    train_loader = create_dataloader(train_set, config['dataloader'].get('train', config['dataloader']), len(args.gpus))
    valid_loader = create_dataloader(valid_set, config['dataloader'].get('valid', config['dataloader']))
    
    trainer = create_trainer(config, model, train_loader, valid_loader)
    
    # ADDED: Resume full training state if checkpoint provided
    if resume_checkpoint and os.path.isfile(resume_checkpoint):
        print_log(f'Resuming training from checkpoint: {resume_checkpoint}')
        checkpoint = torch.load(resume_checkpoint, map_location='cpu')
        
        # Handle both old format (just model) and new format (dict with training state)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format with full training state
            model_state = checkpoint['model_state_dict']
            if args.local_rank != -1:
                # For DDP, load into module
                trainer.model.module.load_state_dict(model_state)
            else:
                trainer.model.load_state_dict(model_state)
            
            # Restore optimizer state
            if 'optimizer_state_dict' in checkpoint and trainer.optimizer is not None:
                trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print_log('Restored optimizer state')
            
            # Restore scheduler state
            if 'scheduler_state_dict' in checkpoint and trainer.scheduler is not None:
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print_log('Restored scheduler state')
            
            # Restore training state
            if 'epoch' in checkpoint:
                trainer.epoch = checkpoint['epoch'] + 1  # Start from next epoch
                print_log(f'Resuming from epoch {trainer.epoch}')
            
            if 'global_step' in checkpoint:
                trainer.global_step = checkpoint['global_step']
                print_log(f'Resuming from step {trainer.global_step}')
            
            if 'best_metric' in checkpoint:
                trainer.best_metric = checkpoint['best_metric']
            
            if 'patience' in checkpoint:
                trainer.patience = checkpoint['patience']
        else:
            # Old format: just model weights
            print_log('Warning: Checkpoint contains only model weights, not full training state')
            if hasattr(checkpoint, 'state_dict'):
                model_state = checkpoint.state_dict()
            else:
                model_state = checkpoint
            
            if args.local_rank != -1:
                trainer.model.module.load_state_dict(model_state)
            else:
                trainer.model.load_state_dict(model_state)
    
    trainer.train(args.gpus, args.local_rank)

if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(args.seed)
    main(args, opt_args)