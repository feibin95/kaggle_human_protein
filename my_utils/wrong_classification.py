import os
import time
import json
import torch
import random
import warnings
import torchvision
import numpy as np
import pandas as pd
import pathlib

from utils import *
from data import HumanDataset
from data import process_df
from data import process_submission_leakdata_full
from data import process_loss_weight
from data import process_together_labels
from tqdm import tqdm
from config import config
from datetime import datetime
from models.model import *
from torch import nn, optim
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score
from PIL import Image
import matplotlib.pyplot as plt

# 1. set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')

index_class_dict = {
    0: "Nucleoplasm",
    1: "Nuclear membrane",
    2: "Nucleoli",
    3: "Nucleoli fibrillar center",
    4: "Nuclear speckles",
    5: "Nuclear bodies",
    6: "Endoplasmic reticulum",
    7: "Golgi apparatus",
    8: "Peroxisomes",
    9: "Endosomes",
    10: "Lysosomes",
    11: "Intermediate filaments",
    12: "Actin filaments",
    13: "Focal adhesion sites",
    14: "Microtubules",
    15: "Microtubule ends",
    16: "Cytokinetic bridge",
    17: "Mitotic spindle",
    18: "Microtubule organizing center",
    19: "Centrosome",
    20: "Lipid droplets",
    21: "Plasma membrane",
    22: "Cell junctions",
    23: "Mitochondria",
    24: "Aggresome",
    25: "Cytosol",
    26: "Cytoplasmic bodies",
    27: "Rods & rings"
}


def check(check_loader, model, folds, val_data_list):
    model.cuda()
    model.eval()
    count = 0
    wrong_id = []
    wrong_class = []
    true_target = []
    wrong_target = []
    true_name = []
    wrong_name = []
    pred = []
    for i, (image, target) in enumerate(tqdm(check_loader)):
        with torch.no_grad():
            image = image.cuda(non_blocking=True)
            y_pred = model(image)
            label = y_pred.sigmoid().cpu().data.numpy()
            label_orig = label.copy().reshape((-1))
            ll = label.copy().reshape((-1))
            ll = -ll
            ll.sort()
            ll = -ll
            threshold = config.threshold
            # if threshold < ll[3]:
            #     threshold = ll[3]
            label = label >= threshold
            label = label.reshape(-1)
            target = np.array(target)
            target = target.reshape(-1)
            flag = True
            for j in range(len(label)):
                if label[j] != target[j]:
                    flag = False
                    break
            if not flag or flag:
                count += 1
                name = val_data_list.iloc[i].Id
                wrong_img_path = os.path.join(config.train_data, name)
                target1 = ' '.join(list([str(k) for k in np.nonzero(target)]))
                label1 = ' '.join(list([str(k) for k in np.nonzero(label)]))
                label1_name = '-&-'.join(list([index_class_dict[k] for k in np.nonzero(label)]))
                label_orig = ' '.join(list(str('%1.2f' % k) for k in label_orig))
                wrong_id.append(str(name))
                wrong_class.append(str(flag))
                true_target.append(target1)
                wrong_target.append(label1)
                pred.append(label_orig)

                images = np.zeros(shape=(512, 512, 3), dtype=np.float)
                r = Image.open(wrong_img_path + "_red.png")
                g = Image.open(wrong_img_path + "_green.png")
                b = Image.open(wrong_img_path + "_blue.png")
                y = Image.open(wrong_img_path + "_yellow.png")
                images[:, :, 0] = np.array(r) / 2 + np.array(y) / 2
                images[:, :, 1] = g
                images[:, :, 2] = b
                images = images.astype(np.uint8)
                f0 = plt.figure(0, figsize=(20, 25))
                f0.suptitle('%s    True:%s     Pred:%s     Pred_name%s' % (str(flag), target1, label1, label1_name))
                ax1 = plt.subplot2grid((5, 4), (0, 0), fig=f0)
                ax2 = plt.subplot2grid((5, 4), (0, 1))
                ax3 = plt.subplot2grid((5, 4), (0, 2))
                ax4 = plt.subplot2grid((5, 4), (0, 3))
                ax5 = plt.subplot2grid((5, 4), (1, 0), rowspan=4, colspan=4)
                ax1.imshow(np.array(r), cmap="Reds")
                ax1.set_title("true:")
                ax2.imshow(np.array(g), cmap="Greens")
                ax2.set_title("pred:")
                ax3.imshow(np.array(b), cmap="Blues")
                ax4.imshow(np.array(y), cmap="Oranges")
                ax5.imshow(images)
                plt.waitforbuttonpress(0)
                plt.close(f0)
    if wrong_id is not []:
        df = pd.DataFrame({
            'Id': wrong_id,
            'True': wrong_class,
            'True_Target': true_target,
            'Pred_Target': wrong_target,
            'pred': pred
        })
        df.to_csv('wrong_classification.csv')


def main():
    fold = config.fold
    model = get_net()
    model.cuda()
    best_model = torch.load(
        "%s/%s_fold_%s_model_best_%s.pth.tar" % (config.best_models, config.model_name, str(fold), config.best))
    model.load_state_dict(best_model["state_dict"])

    train_files = pd.read_csv(config.train_csv)
    external_files = pd.read_csv(config.external_csv)
    test_files = pd.read_csv(config.test_csv)
    all_files, test_files, weight_log = process_df(train_files, external_files, test_files)
    train_data_list, val_data_list = train_test_split(all_files, test_size=0.13, random_state=2050)
    val_data_list = val_data_list[val_data_list['is_external'] == 0]

    check_gen = HumanDataset(val_data_list, augument=False, mode="train")
    check_loader = DataLoader(check_gen, 1, shuffle=False, pin_memory=True, num_workers=6)

    check(check_loader, model, fold, val_data_list)


if __name__ == "__main__":
    main()
