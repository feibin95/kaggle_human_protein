import os 
import time 
import json 
import torch 
import random 
import warnings
import torchvision
import numpy as np 
import pandas as pd 

from utils import *
from data import HumanDataset
from data import process_df
from data import process_submission_leakdata_full
from data import process_dup_img_in_test
from data import process_loss_weight
from data import process_together_labels
from data import tra_val_split
from data import get_threshold
from tqdm import tqdm 
from config import config
from datetime import datetime
from models.model import*
from torch import nn,optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score
from torchvision import transforms as T
# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

if not os.path.exists("./logs/"):
    os.mkdir("./logs/")

log = Logger()
log.open("logs/%s_log_train.txt"%config.model_name,mode="a")
log.write("\n----------------------------------------------- [START %s] %s\n\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
log.write('                           |------------ Train -------------|----------- Valid -------------|----------Best Results---------|------------|\n')
log.write('mode     iter     epoch    |         loss   f1_macro        |         loss   f1_macro       |         loss   f1_macro       | time       |\n')
log.write('-------------------------------------------------------------------------------------------------------------------------------\n')


def train(train_loader, model, criterion, optimizer, epoch, valid_loss, best_results, start, lr):
    losses = AverageMeter()
    model.train()
    # all_target = np.zeros(0)
    # all_pred = np.zeros(0)
    all_target = []
    all_pred = []
    for i,(images,target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        output = model(images)
        target1 = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
        loss = criterion(output,target1)
        losses.update(loss.item(),images.size(0))
        pred = output.sigmoid().cpu().data.numpy() > config.threshold
        all_target.extend(np.array(target).tolist())
        all_pred.extend(pred.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('\r',end='',flush=True)
        message = '%s %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s     %.7f' % (\
                "train", i/len(train_loader) + epoch, epoch,
                losses.avg, 0,
                valid_loss[0], valid_loss[1],
                str(best_results[0])[:8],str(best_results[1])[:8],
                time_to_str((timer() - start),'min'), lr)
        print(message , end='',flush=True)
    # f1 = f1_score(all_target, all_pred, average='macro')
    f1 = f1_score(np.array(all_target), np.array(all_pred), average='macro')
    print('\r', end='', flush=True)
    message = '%s %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s     %.7f' % ( \
        "train", 1 + epoch, epoch,
        losses.avg, f1,
        valid_loss[0], valid_loss[1],
        str(best_results[0])[:8], str(best_results[1])[:8],
        time_to_str((timer() - start), 'min'), lr)
    print(message, end='', flush=True)
    log.write("\n")
    return [losses.avg,f1]


# 2. evaluate function
def evaluate(val_loader, model, criterion, epoch, train_loss, best_results, start):
    losses = AverageMeter()
    model.cuda()
    model.eval()
    all_target = []
    all_pred = []
    with torch.no_grad():
        for i, (images,target) in enumerate(val_loader):
            images_var = images.cuda(non_blocking=True)
            output = model(images_var)
            target1 = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)
            loss = criterion(output,target1)
            losses.update(loss.item(),images_var.size(0))
            pred = output.sigmoid().cpu().data.numpy() > config.threshold
            all_target.extend(np.array(target).tolist())
            all_pred.extend(pred.tolist())
            print('\r',end='',flush=True)
            message = '%s   %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                    "val", i/len(val_loader) + epoch, epoch,
                    train_loss[0], train_loss[1],
                    losses.avg, 0,
                    str(best_results[0])[:8],str(best_results[1])[:8],
                    time_to_str((timer() - start),'min'))
            print(message, end='',flush=True)
        f1 = f1_score(np.array(all_target), np.array(all_pred), average='macro')
        print('\r', end='', flush=True)
        message = '%s   %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % ( \
            "val", 1 + epoch, epoch,
            train_loss[0], train_loss[1],
            losses.avg, f1,
            str(best_results[0])[:8], str(best_results[1])[:8],
            time_to_str((timer() - start), 'min'))
        print(message, end='', flush=True)
        log.write("\n")
    return [losses.avg,f1]


def test(test_loader,model,folds,knum):
    test_num = 1
    if config.tta:
        test_num = 4
    sample_submission_df = pd.read_csv(config.test_csv)
    #threshold_get = get_threshold(config.best, 2)*knum
    threshold_get = get_threshold(config.best, folds) * knum
    threshold_get_copy = threshold_get.copy()
    print(threshold_get_copy)
    for testn in range(test_num):
        #3.1 confirm the model converted to cuda
        labels ,submissions= [],[]
        model.cuda()
        model.eval()
        pred_checkpoint_path = "./checkpoints/pred/fold%d_tta%d.pth.tar" % (folds, testn)
        pred_checkpoint = []
        if os.path.exists(pred_checkpoint_path):
            best_pred = torch.load(pred_checkpoint_path)
            pred_checkpoint = best_pred["pred"]
            for i, label in enumerate(tqdm(pred_checkpoint)):
                if config.opt_thres:
                    threshold = threshold_get_copy
                    labels.append(label > threshold)
                else:
                    ll = label.copy().reshape((-1))
                    ll = -ll
                    ll.sort()
                    ll = -ll
                    threshold = config.threshold
                    if threshold < ll[3]:
                        threshold = ll[3]
                    if threshold > ll[0]:
                        threshold = ll[0]
                    labels.append(label >= threshold)
        else:
            for i,(input,filepath) in enumerate(tqdm(test_loader)):
                #3.2 change everything to cuda and get only basename
                filepath = [os.path.basename(x) for x in filepath]
                inputn = np.zeros(shape=(config.img_weight, config.img_height, 4), dtype=np.uint8)
                inputn[:,:,:] = input.squeeze(0).numpy()[:,:,:]
                with torch.no_grad():
                    input = T.Compose([
                                T.ToPILImage(),
                                T.RandomHorizontalFlip(p=testn%2), # when not tta ----> p=0
                                T.RandomVerticalFlip(p=testn//2), # when not tta ----> p=0
                                T.ToTensor(),
                                T.Normalize([0.0789, 0.0529, 0.0546, 0.0814], [0.147, 0.113, 0.157, 0.148])
                            ])(inputn).float().unsqueeze(0)
                    image_var = input.cuda(non_blocking=True)
                    y_pred = model(image_var)
                    label = y_pred.sigmoid().cpu().data.numpy()
                    pred_checkpoint.append(label)
                    if config.opt_thres:
                        threshold = threshold_get_copy
                        labels.append(label > threshold)
                    else:
                        ll = label.copy().reshape((-1))
                        ll = -ll
                        ll.sort()
                        ll = -ll
                        threshold = config.threshold
                        if threshold < ll[3]:
                            threshold = ll[3]
                        if threshold > ll[0]:
                            threshold = ll[0]
                        labels.append(label >= threshold)
            torch.save({"pred":pred_checkpoint}, pred_checkpoint_path)

        for row in np.concatenate(labels):
            subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
            submissions.append(subrow)
        sample_submission_df['Predicted'] = submissions
        # sample_submission_df = process_submission_leakdata_full(sample_submission_df)
        # sample_submission_df = process_together_labels(sample_submission_df)
        # sample_submission_df = process_dup_img_in_test(sample_submission_df)
        sample_submission_df.to_csv('./submit/%s_best_%s_%d_fold_%d_submission%d.csv'%(config.model_name, config.best,int(knum*10), folds, testn), index=None)


def search_thresholds(all_loader, model):
    model.cuda()
    model.eval()
    all_target = []
    all_pred = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(all_loader)):
            images_var = images.cuda(non_blocking=True)
            output = model(images_var)
            pred = output.sigmoid().cpu().data.numpy()
            all_target.extend(np.array(target).tolist())
            all_pred.extend(pred.tolist())

    print(np.array(all_target).shape)
    print(np.array(all_pred).shape)
    thresholds = np.linspace(0, 1, 100)
    test_threshold = 0.5 * np.ones(28)
    best_threshold = np.zeros(28)
    best_val = np.zeros(28)
    for i in range(28):
        for threshold in thresholds:
            test_threshold[i] = threshold
            score = f1_score(np.array(all_target), np.array(all_pred) > test_threshold, average='macro')
            if score > best_val[i]:
                best_threshold[i] = threshold
                best_val[i] = score
        print("Threshold[%d] %0.6f, F1: %0.6f" % (i, best_threshold[i], best_val[i]))
        test_threshold[i] = best_threshold[i]
    print("Best threshold: ")
    print(best_threshold)
    print("Best f1:")
    print(best_val)


# 4. main function
def main():
    fold = config.fold
    # 4.1 mkdirs
    if not os.path.exists(config.submit):
        os.makedirs(config.submit)
    if not os.path.exists(config.weights + config.model_name + os.sep +str(fold)):
        os.makedirs(config.weights + config.model_name + os.sep +str(fold))
    if not os.path.exists(config.best_models):
        os.mkdir(config.best_models)
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    
    # 4.2 get model
    model = get_net()
    model.cuda()
    if config.is_train_after_crash:
        best_model_name = config.weights + config.model_name + os.sep +str(fold-10) + os.sep + "checkpoint.pth.tar"
        best_model = torch.load(best_model_name)
        print(best_model_name)
        model.load_state_dict(best_model["state_dict"])
        best_results = [np.inf,0]
        val_metrics = [np.inf,0]
        best_results[0] = best_model["best_loss"]
        best_results[1] = best_model["best_f1"]
    else:
        best_results = [np.inf, 0]
        val_metrics = [np.inf, 0]
    print(best_results)
    train_files = pd.read_csv(config.train_csv)
    external_files = pd.read_csv(config.external_csv)
    test_files = pd.read_csv(config.test_csv)
    all_files, test_files, weight_log = process_df(train_files, external_files, test_files)
    # train_data_list,val_data_list = train_test_split(all_files,test_size = 0.13,random_state = 2050)
    train_data_list, val_data_list = tra_val_split(all_files)
    print(len(all_files))
    print(len(train_data_list))
    print(len(val_data_list))
    # train_data_list = train_data_list.iloc[np.arange(10000)]
    # val_data_list = val_data_list.iloc[np.arange(1000)]

    # load dataset
    train_gen = HumanDataset(train_data_list,mode="train")
    sampler = WeightedRandomSampler(train_data_list['freq'].values, num_samples=int(len(train_data_list)*config.multiply), replacement=True)
    train_loader = DataLoader(train_gen,batch_size=config.batch_size,drop_last=True,sampler=sampler,pin_memory=True,num_workers=6)
    # train_loader = DataLoader(train_gen,batch_size=config.batch_size,shuffle=True,pin_memory=True,num_workers=6)

    val_gen = HumanDataset(val_data_list,augument=False,mode="train")
    val_loader = DataLoader(val_gen,batch_size=config.batch_size,drop_last=True,shuffle=False,pin_memory=True,num_workers=6)

    test_gen = HumanDataset(test_files,augument=False,mode="test")
    test_loader = DataLoader(test_gen,1,shuffle=False,pin_memory=True,num_workers=6)

    search_gen = HumanDataset(val_data_list,augument=False,mode="train")
    search_loader = DataLoader(search_gen,batch_size=config.batch_size*4,drop_last=False,shuffle=False,pin_memory=True,num_workers=6)

    # optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4, amsgrad=True)
    optimizer = optim.SGD(model.parameters(),lr = config.lr,momentum=0.9,weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss().cuda()
    # criterion = nn.BCEWithLogitsLoss(torch.from_numpy(process_loss_weight(weight_log)).float()).cuda()
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=4e-8)
    # scheduler = lr_scheduler.StepLR(optimizer,step_size=6,gamma=0.1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, threshold=1e-3)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[6, 13, 20], gamma=0.1)
    start = timer()

    # train
    if config.is_train:
        for epoch in range(0,config.epochs):
            scheduler.step(epoch)
            # train
            lr = get_learning_rate(optimizer)
            train_metrics = train(train_loader,model,criterion,optimizer,epoch,val_metrics,best_results,start,lr)
            # val
            val_metrics = evaluate(val_loader,model,criterion,epoch,train_metrics,best_results,start)
            # check results
            is_best_loss = val_metrics[0] < best_results[0]
            best_results[0] = min(val_metrics[0],best_results[0])
            is_best_f1 = val_metrics[1] > best_results[1]
            best_results[1] = max(val_metrics[1],best_results[1])
            # scheduler.step(val_metrics[0])
            # save model
            save_checkpoint({
                        "epoch":epoch + 1,
                        "model_name":config.model_name,
                        "state_dict":model.state_dict(),
                        "best_loss":best_results[0],
                        "optimizer":optimizer.state_dict(),
                        "fold":fold,
                        "best_f1":best_results[1],
            },is_best_loss,is_best_f1,fold)
            # print logs
            print('\r',end='',flush=True)
            log.write('%s  %5.1f %6.1f         |         %0.3f  %0.3f           |         %0.3f  %0.4f         |         %s  %s    | %s' % (\
                    "best", epoch + 1, epoch + 1,
                    train_metrics[0], train_metrics[1],
                    val_metrics[0], val_metrics[1],
                    str(best_results[0])[:8],str(best_results[1])[:8],
                    time_to_str((timer() - start),'min'))
                )
            log.write("\n")
            time.sleep(0.01)

    if config.is_search_thres:
        best_model_name = "%s/%s_fold_%s_model_best_%s.pth.tar" % (config.best_models, config.model_name, str(fold), config.best)
        # best_model_name = config.weights + config.model_name + os.sep +str(fold) + os.sep + "checkpoint.pth.tar"
        print(best_model_name)
        best_model = torch.load(best_model_name)
        model.load_state_dict(best_model["state_dict"])
        search_thresholds(search_loader, model)

    if config.is_test:
        knums = config.threshold_factor
        for knum in knums:
            for f in range(5):
                best_model_name = "%s/%s_fold_%s_model_best_%s.pth.tar" % (
                config.best_models, config.model_name, str(fold + f), config.best)
                # best_model_name = config.weights + config.model_name + os.sep +str(fold) + os.sep + "checkpoint.pth.tar"
                print(best_model_name)
                best_model = torch.load(best_model_name)
                model.load_state_dict(best_model["state_dict"])
                test(test_loader, model, (fold + f), knum)


if __name__ == "__main__":
    main()
