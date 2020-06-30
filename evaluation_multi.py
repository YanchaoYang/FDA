import json
import os.path as osp
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
from data import CreateTrgDataLoader
from model import CreateModel
import os
from options.test_options import TestOptions
import scipy.io as sio

# color coding of semantic classes
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def fast_hist(a, b, n):
    k = (a>=0) & (a<n)
    return np.bincount( n*a[k].astype(int)+b[k], minlength=n**2 ).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / ( hist.sum(1)+hist.sum(0)-np.diag(hist) )

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[ input==mapping[ind][0] ] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def compute_mIoU( gt_dir, pred_dir, devkit_dir='', restore_from='' ):
    with open( osp.join(devkit_dir, 'info.json'),'r' ) as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)

    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array( info['label2train'],dtype=np.int )
    hist = np.zeros( (num_classes, num_classes) )

    image_path_list = osp.join( devkit_dir, 'val.txt')
    label_path_list = osp.join( devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [osp.join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [osp.join(pred_dir, x.split('/')[-1]) for x in pred_imgs]
    for ind in range(len(gt_imgs)):
        pred  = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format( len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind] ))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            with open(restore_from+'_mIoU.txt', 'a') as f:
                f.write( '{:d} / {:d}: {:0.2f}\n'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))) )
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    hist2 = np.zeros((19, 19))

    for i in range(19):
        hist2[i] = hist[i]/np.sum(hist[i])

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        with open(restore_from+'_mIoU.txt', 'a') as f:
            f.write('===>'+name_classes[ind_class]+':\t' + str(round(mIoUs[ind_class]*100,2)) + '\n')
        print('===>'+name_classes[ind_class]+':\t' + str(round(mIoUs[ind_class]*100,2)))
    with open(restore_from+'_mIoU.txt', 'a') as f:
        f.write('===> mIoU: ' + str(round(np.nanmean(mIoUs)*100,2)) + '\n')

    print('===> mIoU19: ' + str(round(   np.nanmean(mIoUs)*100,2   )))
    print('===> mIoU16: ' + str(round(   np.mean(mIoUs[[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]])*100,2   )))
    print('===> mIoU13: ' + str(round(   np.mean(mIoUs[[0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]])*100,2   )))


def main():
    opt = TestOptions()
    args = opt.initialize()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    args.restore_from = args.restore_opt1
    model1 = CreateModel(args)
    model1.eval()
    model1.cuda()

    args.restore_from = args.restore_opt2
    model2 = CreateModel(args)
    model2.eval()
    model2.cuda()

    args.restore_from = args.restore_opt3
    model3 = CreateModel(args)
    model3.eval()
    model3.cuda()

    targetloader = CreateTrgDataLoader(args)

    # change the mean for different dataset other than CS
    IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
    IMG_MEAN = torch.reshape( torch.from_numpy(IMG_MEAN), (1,3,1,1)  )
    mean_img = torch.zeros(1, 1)

    # ------------------------------------------------- #
    # compute scores and save them
    with torch.no_grad():
        for index, batch in enumerate(targetloader):
            if index % 100 == 0:
                print( '%d processd' % index )
            image, _, name = batch                              # 1. get image
            # create mean image
            if mean_img.shape[-1] < 2:
                B, C, H, W = image.shape
                mean_img = IMG_MEAN.repeat(B,1,H,W)             # 2. get mean image
            image = image.clone() - mean_img                    # 3, image - mean_img
            image = Variable(image).cuda()
 
            # forward
            output1 = model1(image)
            output1 = nn.functional.softmax(output1, dim=1)

            output2 = model2(image)
            output2 = nn.functional.softmax(output2, dim=1)

            output3 = model3(image)
            output3 = nn.functional.softmax(output3, dim=1)

            a, b = 0.3333, 0.3333
            output = a*output1 + b*output2 + (1.0-a-b)*output3

            output = nn.functional.interpolate(output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
            #output = nn.functional.upsample(   output, (1024, 2048), mode='bilinear', align_corners=True).cpu().data[0].numpy()
            output = output.transpose(1,2,0)

            output_nomask = np.asarray( np.argmax(output, axis=2), dtype=np.uint8 )
            output_col = colorize_mask(output_nomask)
            output_nomask = Image.fromarray(output_nomask)    
            name = name[0].split('/')[-1]
            output_nomask.save(  '%s/%s' % (args.save, name)  )
            output_col.save(  '%s/%s_color.png' % (args.save, name.split('.')[0])  ) 
    # scores computed and saved
    # ------------------------------------------------- #
    compute_mIoU( args.gt_dir, args.save, args.devkit_dir, args.restore_from )    


if __name__ == '__main__':
    main()

