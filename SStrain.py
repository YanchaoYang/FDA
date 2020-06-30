import torch.nn.functional as F
import numpy as np
from options.train_options import TrainOptions
from utils.timer import Timer
import os
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from model import CreateModel
import torch.backends.cudnn as cudnn
import torch
from torch.autograd import Variable
from utils import FDA_source_to_target
import scipy.io as sio
from data import CreatePseudoTrgLoader

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
CS_weights = np.array( (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=np.float32 )
CS_weights = torch.from_numpy(CS_weights)

def main():
    opt = TrainOptions()
    args = opt.initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    _t = {'iter time' : Timer()}

    model_name = args.source + '_to_' + args.target
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
        os.makedirs(os.path.join(args.snapshot_dir, 'logs'))
    opt.print_options(args)

    sourceloader, targetloader = CreateSrcDataLoader(args), CreateTrgDataLoader(args)
    sourceloader_iter, targetloader_iter = iter(sourceloader), iter(targetloader)

    pseudotrgloader = CreatePseudoTrgLoader(args)
    pseudoloader_iter = iter(pseudotrgloader)

    model, optimizer = CreateModel(args)

    start_iter = 0
    if args.restore_from is not None:
        start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[1])

    cudnn.enabled = True
    cudnn.benchmark = True

    model.train()
    model.cuda()

    # losses to log
    loss = ['loss_seg_src', 'loss_seg_psu']
    loss_train = 0.0
    loss_val = 0.0
    loss_pseudo = 0.0
    loss_train_list = []
    loss_val_list = []
    loss_pseudo_list = []

    mean_img = torch.zeros(1, 1)
    class_weights = Variable(CS_weights).cuda()

    _t['iter time'].tic()
    for i in range(start_iter, args.num_steps):
        model.adjust_learning_rate(args, optimizer, i)                               # adjust learning rate
        optimizer.zero_grad()                                                        # zero grad

        src_img, src_lbl, _, _ = sourceloader_iter.next()                            # new batch source
        trg_img, trg_lbl, _, _ = targetloader_iter.next()                            # new batch target
        psu_img, psu_lbl, _, _ = pseudoloader_iter.next()

        scr_img_copy = src_img.clone()

        if mean_img.shape[-1] < 2:
            B, C, H, W = src_img.shape
            mean_img = IMG_MEAN.repeat(B,1,H,W)

        #-------------------------------------------------------------------#

        # 1. source to target, target to target
        src_in_trg = FDA_source_to_target( src_img, trg_img, L=args.LB )            # src_lbl
        trg_in_trg = trg_img

        # 2. subtract mean
        src_img = src_in_trg.clone() - mean_img                                 # src_1, trg_1, src_lbl
        trg_img = trg_in_trg.clone() - mean_img                                 # trg_1, trg_0, trg_lbl
        psu_img = psu_img.clone()    - mean_img

        #-------------------------------------------------------------------#

        # evaluate and update params #####
        src_img, src_lbl = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda() # to gpu
        src_seg_score = model(src_img, lbl=src_lbl, weight=class_weights, ita=args.ita)      # forward pass
        loss_seg_src = model.loss_seg                                                # get loss
        loss_ent_src = model.loss_ent

        # use pseudo label as supervision
        psu_img, psu_lbl = Variable(psu_img).cuda(), Variable(psu_lbl.long()).cuda()
        psu_seg_score = model(psu_img, lbl=psu_lbl, weight=class_weights, ita=args.ita)
        loss_seg_psu = model.loss_seg
        loss_ent_psu = model.loss_ent

        loss_all = loss_seg_src + ( loss_seg_psu + args.entW*loss_ent_psu )    # loss of seg on src, and ent on s and t
        loss_all.backward()
        optimizer.step()

        loss_train += loss_seg_src.detach().cpu().numpy()
        loss_val   += loss_seg_psu.detach().cpu().numpy()
        
        if (i+1) % args.save_pred_every == 0:
            print('taking snapshot ...')
            torch.save( model.state_dict(), os.path.join(args.snapshot_dir, '%s_' % (args.source) + str(i+1) + '.pth') )
            
        if (i+1) % args.print_freq == 0:
            _t['iter time'].toc(average=False)
            print('[it %d][src seg loss %.4f][psu seg loss %.4f][lr %.4f][%.2fs]' % \
                    (i + 1, loss_seg_src.data, loss_seg_psu.data, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff) )

            sio.savemat(args.tempdata, {'src_img':src_img.cpu().numpy(), 'trg_img':trg_img.cpu().numpy()})

            loss_train /= args.print_freq
            loss_val   /= args.print_freq
            loss_train_list.append(loss_train)
            loss_val_list.append(loss_val)
            sio.savemat( args.matname, {'loss_train':loss_train_list, 'loss_val':loss_val_list} )
            loss_train = 0.0
            loss_val = 0.0

            if i + 1 > args.num_steps_stop:
                print('finish training')
                break
            _t['iter time'].tic()

if __name__ == '__main__':
    main()

