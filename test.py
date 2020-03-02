import os
import torch

import parser
import models
import data

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mean_iou_evaluate import mean_iou_score, read_masks
import torchvision.utils
from PIL import Image


from sklearn.metrics import accuracy_score

def evaluate(model, data_loader):

    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad(): # do not need to caculate information for gradient during eval
        for idx, (imgs, gt, filename) in enumerate(data_loader):

            # imgs = imgs.cuda()

            pred = model(imgs)
            
            _, pred = torch.max(pred, dim = 1)

            print(pred.shape)
            pred = pred.cpu().numpy().squeeze()
            # gt = gt.numpy().squeeze()

            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    return mean_iou_score(preds, gts, 9)


def evaluate_save(model, data_loader):
    ''' set model to evaluate mode '''
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs, gt, filename) in enumerate(data_loader):
            # imgs = imgs.cuda()

            pred = model(imgs)

            _, pred = torch.max(pred, dim=1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()

            save_imgs(pred, filename)


def save_imgs(imglist, filename):

    n = 0
    for i in imglist:
        # torchvision.utils.save_image(i, "./outputfiles/" + filename[n])
        print(set(i.flatten()))
        exit()
        result = Image.fromarray((i).astype(np.uint8))
        result.save( "./outputfiles/" + filename[n])
        n +=1


if __name__ == '__main__':

    
    args = parser.arg_parse()

    ''' setup GPU '''
    # torch.cuda.set_device(args.gpu)

    ''' prepare data_loader '''
    print('===> prepare data loader ...')
    test_loader = torch.utils.data.DataLoader(data.DataLoaderSegmentation(args, mode='val'),
                                              batch_size=args.test_batch,
                                              num_workers=args.workers,
                                              shuffle=True)
    ''' prepare mode '''
    model = models.Net(args)\
        # .cuda()
    model_std = torch.load(os.path.join(args.save_dir, 'model_best.pth.tar'), map_location = "cpu")
    model.load_state_dict(model_std)

    ''' resume save model '''
    checkpoint = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(checkpoint)

    acc = evaluate(model, test_loader)
    print('Testing Accuracy: {}'.format(acc))
