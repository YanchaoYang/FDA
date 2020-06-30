import argparse
import os.path as osp

class TrainOptions():
    def initialize(self):
        parser = argparse.ArgumentParser( description="training script for FDA" )
        parser.add_argument("--model", type=str, default='DeepLab', help="available options : DeepLab and VGG")
        parser.add_argument("--LB", type=float, default=0.1, help="beta for FDA")
        parser.add_argument("--GPU", type=str, default='0', help="which GPU to use")
        parser.add_argument("--entW", type=float, default=0.005, help="weight for entropy")
        parser.add_argument("--ita", type=float, default=2.0, help="ita for robust entropy")
        parser.add_argument("--switch2entropy", type=int, default=50000, help="switch to entropy after this many steps")

        parser.add_argument("--source", type=str, default='gta5', help="source dataset : gta5 or synthia")
        parser.add_argument("--target", type=str, default='cityscapes', help="target dataset : cityscapes")
        parser.add_argument("--snapshot-dir", type=str, default='../checkpoints/FDA', help="Where to save snapshots of the model.")
        parser.add_argument("--data-dir", type=str, default='../data_semseg/GTA5', help="Path to the directory containing the source dataset.")
        parser.add_argument("--data-list", type=str, default='./dataset/gta5_list/train.txt', help="Path to the listing of images in the source dataset.")
        parser.add_argument("--data-dir-target", type=str, default='../data_semseg/cityscapes', help="Path to the directory containing the target dataset.")
        parser.add_argument("--data-list-target", type=str, default='./dataset/cityscapes_list/train.txt', help="list of images in the target dataset.")
        parser.add_argument("--set", type=str, default='train', help="choose adaptation set.")
        parser.add_argument("--label-folder", type=str, default=None, help="Path to the directory containing the pseudo labels.")

        parser.add_argument("--batch-size", type=int, default=1, help="input batch size.")
        parser.add_argument("--num-steps", type=int, default=150000, help="Number of training steps.")
        parser.add_argument("--num-steps-stop", type=int, default=100000, help="Number of training steps for early stopping.")
        parser.add_argument("--num-workers", type=int, default=4, help="number of threads.")
        parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="initial learning rate for the segmentation network.")
        parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")
        parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")
        parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate (only for deeplab).")

        parser.add_argument("--num-classes", type=int, default=19, help="Number of classes for cityscapes.")
        parser.add_argument("--init-weights", type=str, default=None, help="initial model.")
        parser.add_argument("--restore-from", type=str, default=None, help="Where restore model parameters from.")

        parser.add_argument("--save-pred-every", type=int, default=2500, help="Save summaries and checkpoint every often.")
        parser.add_argument("--print-freq", type=int, default=100, help="print loss and time fequency.")
        parser.add_argument("--matname", type=str, default='loss_log.mat', help="mat name to save loss")
        parser.add_argument("--tempdata", type=str, default='tempdata.mat', help="mat name to save data")

        return parser.parse_args()
    
    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
    
        # save to the disk
        file_name = osp.join(args.snapshot_dir, 'opt.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')    

