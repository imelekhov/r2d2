# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb
import torch
import torch.optim as optim
from os import path as osp

from tools import common
from tools.trainer import Trainer
from tools.scheduler import WarmupScheduler
from tools.dataloader import *
from nets.patchnet import *
from nets.losses import *

default_net = "Quad_L2Net_ConfCFS()"

toy_db_debug = """SyntheticPairDataset(
    ImgFolder('imgs'), 
            'RandomScale(256,1024,can_upscale=True)', 
            'RandomTilting(0.5), PixelNoise(25)')"""

db_web_images = """SyntheticPairDataset(
    web_images, 
        'RandomScale(256,1024,can_upscale=True)',
        'RandomTilting(0.5), PixelNoise(25)')"""

db_aachen_images = """SyntheticPairDataset(
    aachen_db_images, 
        'RandomScale(256,1024,can_upscale=True)', 
        'RandomTilting(0.5), PixelNoise(25)')"""

db_aachen_style_transfer = """TransformedPairs(
    aachen_style_transfer_pairs,
            'RandomScale(256,1024,can_upscale=True), RandomTilting(0.5), PixelNoise(25)')"""

db_phototourism_train = """SyntheticPairDataset(
    phototourism_dataset_train, 
        'RandomScale(256,1024,can_upscale=True)',
        'RandomTilting(0.5), PixelNoise(25)')"""

db_phototourism_val = """SyntheticPairDataset(
    phototourism_dataset_val, 
        'RandomScale(256,1024,can_upscale=True)',
        'RandomTilting(0.5), PixelNoise(25)')"""

db_phototourism_style_train = """TransformedPairs(
    phototourism_style_dataset_train,
            'RandomScale(256,1024,can_upscale=True), RandomTilting(0.5), PixelNoise(25)')"""

db_phototourism_style_val = """TransformedPairs(
    phototourism_style_dataset_val,
            'RandomScale(256,1024,can_upscale=True), RandomTilting(0.5), PixelNoise(25)')"""

db_aachen_flow = "aachen_flow_pairs"

data_sources = dict(
    D = toy_db_debug,
    W = db_web_images,
    A = db_aachen_images,
    F = db_aachen_flow,
    S = db_aachen_style_transfer,
    P = db_phototourism_train,
    X = db_phototourism_style_train,
    )

default_dataloader = """PairLoader(CatPairDataset(`data`),
    scale   = 'RandomScale(256,1024,can_upscale=True)',
    distort = 'ColorJitter(0.2,0.2,0.2,0.1)',
    crop    = 'RandomCrop(192)',
    do_color_aug=True)"""

val_dataloader = """PairLoader(CatPairDataset(`data`),
    scale   = 'RandomScale(256,1024,can_upscale=True)',
    crop    = 'RandomCrop(192)')"""

default_sampler = """NghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5, border=16,
                            subd_neg=-8,maxpool_pos=True)"""

default_loss = """MultiLoss(
        1, ReliabilityLoss(`sampler`, base=0.5, nq=20),
        1, CosimLoss(N=`N`),
        1, PeakyLoss(N=`N`))"""

'''
class MyTrainer(trainer.Trainer):
    """ This class implements the network training.
        Below is the function I need to overload to explain how to do the backprop.
    """
    def train(self, inputs):
        output = self.net(imgs=[inputs.pop('img1'),inputs.pop('img2')])
        allvars = dict(inputs, **output)
        loss, details = self.loss_func(**allvars)
        if torch.is_grad_enabled(): loss.backward()
        return loss, details
'''


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Train R2D2")

    parser.add_argument("--data-loader", type=str, default=default_dataloader)
    parser.add_argument("--train-data", type=str, default=list('WASF'), nargs='+', 
        choices = set(data_sources.keys()))
    parser.add_argument("--net", type=str, default=default_net, help='network architecture')

    parser.add_argument("--pretrained", type=str, default="", help='pretrained model path')
    parser.add_argument("--save-path", type=str, required=True, help='model save_path path')
    
    parser.add_argument("--loss", type=str, default=default_loss, help="loss function")
    parser.add_argument("--sampler", type=str, default=default_sampler, help="AP sampler")
    parser.add_argument("--N", type=int, default=16, help="patch size for repeatability")

    parser.add_argument("--epochs", type=int, default=70, help='number of training epochs')
    parser.add_argument("--batch-size", "--bs", type=int, default=7, help="batch size")
    # parser.add_argument("--learning-rate", "--lr", type=str, default=1e-4)
    parser.add_argument("--learning-rate", "--lr", type=str, default=1e-3)
    parser.add_argument("--weight-decay", "--wd", type=float, default=5e-4)
    
    parser.add_argument("--threads", type=int, default=16, help='number of worker threads')
    parser.add_argument("--gpu", type=int, nargs='+', default=[1], help='-1 for CPU')
    
    args = parser.parse_args()
    
    iscuda = common.torch_set_gpu(args.gpu)
    common.mkdir_for(args.save_path)

    # Create data loader
    from datasets import *
    db_train = [data_sources[key] for key in args.train_data]
    db_train = eval(args.data_loader.replace('`data`',','.join(db_train)).replace('\n',''))
    db_val = [db_phototourism_val if args.train_data[0] == 'P' else db_phototourism_style_val]
    db_val = eval(val_dataloader.replace('`data`', ','.join(db_val)).replace('\n', ''))
    print("Training image database =", db_train)
    train_loader = threaded_loader(db_train, iscuda, args.threads, args.batch_size, shuffle=True)
    val_loader = threaded_loader(db_val, iscuda, args.threads, args.batch_size, shuffle=False)

    # create network
    print("\n>> Creating net = " + args.net) 
    net = eval(args.net)
    print(f" ( Model size: {common.model_size(net)/1000:.0f}K parameters )")

    # initialization
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, lambda a,b:a)
        net.load_pretrained(checkpoint['state_dict'])
        
    # create losses
    loss = args.loss.replace('`sampler`',args.sampler).replace('`N`',str(args.N))
    print("\n>> Creating loss = " + loss)
    loss = eval(loss.replace('\n',''))
    
    # create optimizer
    optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad],
                           lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = WarmupScheduler(optimizer,
                                total_epochs=args.epochs,
                                warmup_for=5,
                                min_lr=args.learning_rate / 60.)

    trainer = Trainer(net, train_loader, val_loader, loss, optimizer, scheduler)
    if iscuda: trainer = trainer.cuda()

    # Training loop #
    max_val_error = 1e5
    for epoch in range(args.epochs):
        print(f"\n>> Starting epoch {epoch}...")
        trainer.train()

        val_error = trainer.validate()
        if val_error < max_val_error:
            print(f"\n>> Saving model (the lowest validation error)")
            torch.save({'net': args.net, 'state_dict': net.state_dict()},
                       osp.join(args.save_path, "min_val_error_checkpoint.pt"))
            max_val_error = val_error

        torch.save({'net': args.net, 'state_dict': net.state_dict()},
                   osp.join(args.save_path, "checkpoint_" + str(epoch) + ".pt"))
        '''
        if epoch == 30:
            print(f"\n>> Saving model (30th epoch)")
            torch.save({'net': args.net, 'state_dict': net.state_dict()},
                       osp.join(args.save_path, "checkpoint_30th.pt"))
        '''

    print(f"\n>> Saving model to {args.save_path}")
    torch.save({'net': args.net, 'state_dict': net.state_dict()},
               osp.join(args.save_path, "end_train_checkpoint.pt"))



