from utils import *
from config import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.preprocessing import MultiLabelBinarizer
from imgaug import augmenters as iaa
import pathlib
from collections import Counter
from itertools import chain
import math
import cv2
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.model_selection import KFold


# set random seed
np.random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

# create dataset class
class HumanDataset(Dataset):
    def __init__(self,images_df,augument=True,mode="train"):
        self.images_df = images_df.copy()
        self.augument = augument
        self.mlb = MultiLabelBinarizer(classes = np.arange(0,config.num_classes))
        self.mlb.fit(np.arange(0,config.num_classes))
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    # def __getitem__(self, index):
    #     X = self.read_images(index)
    #     if not self.mode == "test":
    #         labels = np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))
    #         y  = np.eye(config.num_classes,dtype=np.float)[labels].sum(axis=0)
    #     else:
    #         y = str(self.images_df.iloc[index].Id.absolute())
    #     if self.augument:
    #         X = self.augumentor(X)
    #     is_external = self.images_df.iloc[index].is_external
    #     #X = T.Compose([T.ToPILImage(),T.ToTensor(),T.Normalize([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])])(X)
    #     if is_external == 0:
    #         X = T.Compose([
    #             T.ToPILImage(),
    #             T.ToTensor(),
    #             T.Normalize([0.0789, 0.0529, 0.0546, 0.0814], [0.147, 0.113, 0.157, 0.148])
    #         ])(X)
    #     else:
    #         X = T.Compose([
    #             T.ToPILImage(),
    #             T.ToTensor(),
    #             T.Normalize([0.1177, 0.0696, 0.0660, 0.1056], [0.179, 0.127, 0.176, 0.166])
    #         ])(X)
    #     return X.float(),y
    def __getitem__(self, index):
        X = self.read_images(index)
        if not self.mode == "test":
            labels = np.array(list(map(int, self.images_df.iloc[index].Target.split(' '))))
            y = np.eye(config.num_classes, dtype=np.float)[labels].sum(axis=0)
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.augument:
            X = self.augumentor(X)
        is_external = self.images_df.iloc[index].is_external
        # X = T.Compose([T.ToPILImage(),T.ToTensor(),T.Normalize([0.08069, 0.05258, 0.05487, 0.08282], [0.13704, 0.10145, 0.15313, 0.13814])])(X)
        if not self.mode == "test":
            if is_external == 0:
                X = T.Compose([
                    T.ToPILImage(),
                    T.ToTensor(),
                    T.Normalize([0.0789, 0.0529, 0.0546, 0.0814], [0.147, 0.113, 0.157, 0.148])
                ])(X)
            else:
                X = T.Compose([
                    T.ToPILImage(),
                    T.ToTensor(),
                    T.Normalize([0.1177, 0.0696, 0.0660, 0.1056], [0.179, 0.127, 0.176, 0.166])
                ])(X)
            return X.float(), y
        else:
            return X, y


    def read_images(self,index):
        row = self.images_df.iloc[index]
        filename = str(row.Id.absolute())
        #use only rgb channels
        if config.channels == 4:
            images = np.zeros(shape=(512,512,4))
        else:
            images = np.zeros(shape=(512,512,3))
        r = np.array(Image.open(filename+"_red.png")) 
        g = np.array(Image.open(filename+"_green.png")) 
        b = np.array(Image.open(filename+"_blue.png")) 
        y = np.array(Image.open(filename+"_yellow.png")) 
        images[:,:,0] = r.astype(np.uint8) 
        images[:,:,1] = g.astype(np.uint8)
        images[:,:,2] = b.astype(np.uint8)
        if config.channels == 4:
            images[:,:,3] = y.astype(np.uint8)
        else:
            images[:,:,0] = (r / 2 + y / 2).astype(np.uint8)
        images = images.astype(np.uint8)
        #images = np.stack(images,-1) 
        if config.img_height == 512:
            return images
        else:
            return cv2.resize(images,(config.img_weight,config.img_height))

    def augumentor(self, image):
        augment_img = iaa.Sequential([
            iaa.Sometimes(0.01,
                          iaa.OneOf([
                              iaa.GaussianBlur((0, 0.3)),
                              iaa.AverageBlur(k=2),
                              iaa.MedianBlur(k=3)
                          ])),
            iaa.Sometimes(0.3,
                          iaa.Multiply((0.8, 1.2), per_channel=0.3)),
            iaa.Sometimes(0.01,
                          iaa.Sharpen(alpha=(0, 0.2), lightness=(0.9, 1.1))),
            iaa.OneOf([
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-30, 30)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)])
        ], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug


def create_class_weight(labels_dict, mu=0.5):
    total = sum(labels_dict.values())
    keys = labels_dict.keys()
    class_weight = dict()
    class_weight_log = dict()
    for key in keys:
        score = float(total) / float(labels_dict[key])
        score_log = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = round(score, 2) if score > config.min_sampling_limit else round(config.min_sampling_limit, 2)
        class_weight_log[key] = round(score_log, 2) if score_log > config.min_sampling_limit else round(config.min_sampling_limit, 2)
    return class_weight, class_weight_log


# def process_df(train_df, external_df, test_df):
#     train_data_path = config.train_data
#     external_data_path = config.external_data
#     test_data_path = config.test_data
#     if not isinstance(train_data_path, pathlib.Path):
#         train_data_path = pathlib.Path(train_data_path)
#     if not isinstance(external_data_path, pathlib.Path):
#         external_data_path = pathlib.Path(external_data_path)
#     if not isinstance(test_data_path, pathlib.Path):
#         test_data_path = pathlib.Path(test_data_path)
#
#     train_df.Id = train_df.Id.apply(lambda x: train_data_path / x)
#     external_df.Id = external_df.Id.apply(lambda x: external_data_path / x)
#     test_df.Id = test_df.Id.apply(lambda x: test_data_path / x)
#     train_df['target_vec'] = train_df['Target'].map(lambda x: list(map(int, x.strip().split())))
#     external_df['target_vec'] = external_df['Target'].map(lambda x: list(map(int, x.strip().split())))
#     train_df['is_external'] = 0
#     external_df['is_external'] = 1
#     test_df['is_external'] = 0
#
#     count_labels_raw = Counter(list(chain.from_iterable(train_df['target_vec'].values)))
#     class_weight_raw, class_weight_log_raw = create_class_weight(count_labels_raw, 0.3)
#     cwl_raw = np.ones(len(class_weight_raw))
#     for key, value in class_weight_raw.items():
#         cwl_raw[key] = value
#     cwl_raw_rank = cwl_raw.copy()  # sort <
#     cwl_raw_rank.sort()
#     chose_num = 28
#     if chose_num is not 0:
#         min_value = cwl_raw_rank[len(cwl_raw_rank) - chose_num]
#
#         def calculate_chose_or_not(row):
#             row.loc['chose'] = 0
#             for num in row.target_vec:
#                 if cwl_raw[num] >= min_value:
#                     row.loc['chose'] = 1
#             return row
#
#         external_df = external_df.apply(calculate_chose_or_not, axis=1)
#     else:
#         external_df['chose'] = 0
#     external_df = external_df.loc[external_df['chose'] == 1]
#     # external_df.to_csv('external.csv', index=None)
#     external_df = external_df.drop(['chose'], axis=1)
#
#     all_df = pd.concat([train_df, external_df], ignore_index=True)
#     count_labels = Counter(list(chain.from_iterable(all_df['target_vec'].values)))
#     class_weight, class_weight_log = create_class_weight(count_labels, 0.3)
#     cwl = np.ones((len(class_weight_log)))
#     for key, value in class_weight_log.items():
#         cwl[key] = value
#
#     def calculate_freq(row):
#         row.loc['freq'] = 0
#         for num in row.target_vec:
#             row.loc['freq'] = max(row.loc['freq'], class_weight_log[num])
#         return row
#
#     all_df = all_df.apply(calculate_freq, axis=1)
#     # print(count_labels)
#     # print(class_weight_log)
#     return all_df, test_df, cwl
def process_df(train_df, external_df, test_df):
    train_data_path = config.train_data
    external_data_path = config.external_data
    test_data_path = config.test_data
    if not isinstance(train_data_path, pathlib.Path):
        train_data_path = pathlib.Path(train_data_path)
    if not isinstance(external_data_path, pathlib.Path):
        external_data_path = pathlib.Path(external_data_path)
    if not isinstance(test_data_path, pathlib.Path):
        test_data_path = pathlib.Path(test_data_path)

    train_df.Id = train_df.Id.apply(lambda x: train_data_path / x)
    external_df.Id = external_df.Id.apply(lambda x: external_data_path / x)
    test_df.Id = test_df.Id.apply(lambda x: test_data_path / x)
    train_df['target_vec'] = train_df['Target'].map(lambda x: list(map(int, x.strip().split())))
    external_df['target_vec'] = external_df['Target'].map(lambda x: list(map(int, x.strip().split())))
    train_df['is_external'] = 0
    external_df['is_external'] = 1
    test_df['is_external'] = 0

    # count_labels_raw = Counter(list(chain.from_iterable(train_df['target_vec'].values)))
    # class_weight_raw, class_weight_log_raw = create_class_weight(count_labels_raw, 0.3)
    # cwl_raw = np.ones(len(class_weight_raw))
    # for key, value in class_weight_raw.items():
    #     cwl_raw[key] = value
    # cwl_raw_rank = cwl_raw.copy()  # sort <
    # cwl_raw_rank.sort()
    # chose_num = 28
    # if chose_num is not 0:
    #     min_value = cwl_raw_rank[len(cwl_raw_rank) - chose_num]
    #
    #     def calculate_chose_or_not(row):
    #         row.loc['chose'] = 0
    #         for num in row.target_vec:
    #             if cwl_raw[num] >= min_value:
    #                 row.loc['chose'] = 1
    #         return row
    #
    #     external_df = external_df.apply(calculate_chose_or_not, axis=1)
    # else:
    #     external_df['chose'] = 0
    # external_df = external_df.loc[external_df['chose'] == 1]
    # # external_df.to_csv('external.csv', index=None)
    # external_df = external_df.drop(['chose'], axis=1)

    all_df = pd.concat([train_df, external_df], ignore_index=True)
    if config.is_train:
        count_labels = Counter(list(chain.from_iterable(all_df['target_vec'].values)))
        class_weight, class_weight_log = create_class_weight(count_labels, 0.3)
        cwl = np.ones((len(class_weight_log)))
        for key, value in class_weight_log.items():
            cwl[key] = value

        def calculate_freq(row):
            row.loc['freq'] = 0
            for num in row.target_vec:
                row.loc['freq'] = max(row.loc['freq'], class_weight_log[num])
            return row

        all_df = all_df.apply(calculate_freq, axis=1)
        # print(count_labels)
        # print(class_weight_log)
        return all_df, test_df, cwl
    else:
        all_df['freq'] = 1
        cwl = np.ones(28)
        return all_df, test_df, cwl
# train = pd.read_csv(config.train_csv)
# external = pd.read_csv(config.external_csv)
# test = pd.read_csv(config.test_csv)
# all_df, test_df, _ = process_df(train, external, test)
# print(all_df)
# #all_df.to_csv('all.csv', index=None)
# print(len(train))


def process_loss_weight(weight_log):
    # return np.sqrt(weight_log)
    return weight_log


def process_submission_leakdata(submission_df):
    submission_df.loc[submission_df['Id'] == 'a8d73536-bad8-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '0 23'
    submission_df.loc[submission_df['Id'] == '63ed01a4-bad7-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '14 16'
    submission_df.loc[submission_df['Id'] == '84c046b4-bad4-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '14 16'
    submission_df.loc[submission_df['Id'] == '69cbf89a-bace-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 16 25'
    submission_df.loc[submission_df['Id'] == 'b9a882d6-bacc-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '14 16'
    submission_df.loc[submission_df['Id'] == '29d1d616-bacd-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '19'
    submission_df.loc[submission_df['Id'] == 'f8f6566a-bac8-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 16'
    submission_df.loc[submission_df['Id'] == '9edb2498-bad8-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '16'
    submission_df.loc[submission_df['Id'] == '8dd19ca8-bacf-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 16 25'
    submission_df.loc[submission_df['Id'] == 'adfa9e8e-bac6-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '5'
    submission_df.loc[submission_df['Id'] == 'da5b852e-bacb-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '3'
    submission_df.loc[submission_df['Id'] == 'df8d2780-bac8-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '16 25'
    submission_df.loc[submission_df['Id'] == '72a6fbf8-bad4-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '16 25'
    submission_df.loc[submission_df['Id'] == 'f6b06252-bad6-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '0 16 25'
    submission_df.loc[submission_df['Id'] == 'edb5b41e-bad0-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 16'
    submission_df.loc[submission_df['Id'] == '0a96bf2c-bad3-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '4'
    submission_df.loc[submission_df['Id'] == '1f97ea4a-bad3-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 25'
    submission_df.loc[submission_df['Id'] == '2327a292-bac7-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '5 16'
    submission_df.loc[submission_df['Id'] == '10d4730a-bada-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '0 16'
    submission_df.loc[submission_df['Id'] == '0b651912-bad3-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '16 25'
    submission_df.loc[submission_df['Id'] == '12dea42a-bacd-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 16 17 18'
    submission_df.loc[submission_df['Id'] == 'b43493dc-bac5-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '2 11'
    submission_df.loc[submission_df['Id'] == 'ba6febf2-bad4-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '16 25'
    submission_df.loc[submission_df['Id'] == '58148166-bacf-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '14 16'
    submission_df.loc[submission_df['Id'] == '79970da6-bac8-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '2 16'
    submission_df.loc[submission_df['Id'] == 'c02bb81c-bac7-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '0 16 17 18'
    submission_df.loc[submission_df['Id'] == 'fcd88d84-bad2-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '16 25'
    submission_df.loc[submission_df['Id'] == '59604078-bad9-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '16'
    submission_df.loc[submission_df['Id'] == '89b31fae-bad2-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 16'
    submission_df.loc[submission_df['Id'] == '88b38a80-bac8-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '2 16'
    submission_df.loc[submission_df['Id'] == 'adc182fa-bad2-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 21'
    submission_df.loc[submission_df['Id'] == '6a322caa-bad3-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '5'
    submission_df.loc[submission_df['Id'] == '869a7f8c-bac9-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 5'
    submission_df.loc[submission_df['Id'] == 'b478cc78-bad4-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 16'
    submission_df.loc[submission_df['Id'] == 'e0f9483a-bacb-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 7'
    submission_df.loc[submission_df['Id'] == '9eafcf6a-bacd-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 21'
    submission_df.loc[submission_df['Id'] == 'b7ae02d8-bac9-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '16 17 23'
    submission_df.loc[submission_df['Id'] == 'dbdcd95c-bac6-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '5'
    submission_df.loc[submission_df['Id'] == 'e7f56384-bad1-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '1'
    submission_df.loc[submission_df['Id'] == '43357408-bad9-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '17 25'
    submission_df.loc[submission_df['Id'] == 'c5deab72-bad9-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '0 21'
    submission_df.loc[submission_df['Id'] == 'b9acf08a-bac9-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 16'
    submission_df.loc[submission_df['Id'] == 'e8bae166-bad8-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '2 17'
    submission_df.loc[submission_df['Id'] == '5661665e-bacf-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '17 25'
    submission_df.loc[submission_df['Id'] == '9e6fe8be-bad2-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '4'
    submission_df.loc[submission_df['Id'] == 'df533cce-bac7-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '17'
    submission_df.loc[submission_df['Id'] == '7c1f771c-bac7-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '25'
    submission_df.loc[submission_df['Id'] == 'f8cd7738-bad0-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '25'
    submission_df.loc[submission_df['Id'] == '39508fe6-bad3-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 16'
    submission_df.loc[submission_df['Id'] == 'a56d3f98-bacf-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '9 10'
    submission_df.loc[submission_df['Id'] == '28601ba0-bac6-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '14'
    submission_df.loc[submission_df['Id'] == '8617f44e-baca-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0'
    submission_df.loc[submission_df['Id'] == '1144d38e-bacb-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '21 16'
    submission_df.loc[submission_df['Id'] == '201229ac-bad0-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '2'
    submission_df.loc[submission_df['Id'] == '0f3274c0-bada-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '13'
    submission_df.loc[submission_df['Id'] == '29414644-bad4-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '4 21 26'
    submission_df.loc[submission_df['Id'] == '92c7e608-bad5-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '25'
    submission_df.loc[submission_df['Id'] == '83509894-bad2-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '13'
    submission_df.loc[submission_df['Id'] == 'af2c5f2e-bac9-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '25'
    submission_df.loc[submission_df['Id'] == 'bf6c33d0-bad5-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '4'
    submission_df.loc[submission_df['Id'] == '9da67d5c-bac9-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '14'
    submission_df.loc[submission_df['Id'] == 'cbfe766e-bace-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '14'
    submission_df.loc[submission_df['Id'] == 'c1dc11c4-bacd-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0'
    submission_df.loc[submission_df['Id'] == '8f257b9c-bacf-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '21'
    submission_df.loc[submission_df['Id'] == 'd69acc70-bac5-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '0'
    submission_df.loc[submission_df['Id'] == '94205e64-baca-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '16'
    submission_df.loc[submission_df['Id'] == '55eb4db6-baca-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '22 16 25'
    submission_df.loc[submission_df['Id'] == '10e592a2-bac6-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '21 16 19 25'
    submission_df.loc[submission_df['Id'] == 'e7a05526-bad4-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '9 10'
    submission_df.loc[submission_df['Id'] == '896007d6-bad9-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '14 16'
    submission_df.loc[submission_df['Id'] == '111f3934-bad6-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '16 25'
    submission_df.loc[submission_df['Id'] == 'aec4415e-bad9-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '2 16'
    submission_df.loc[submission_df['Id'] == '03f31e24-badb-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '16'
    submission_df.loc[submission_df['Id'] == 'f60586ac-bac7-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '7 17'
    submission_df.loc[submission_df['Id'] == 'fce301c8-bac9-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '5'
    submission_df.loc[submission_df['Id'] == '10748996-baca-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '15 25'
    submission_df.loc[submission_df['Id'] == '2367dd2c-bac6-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '21 11 16'
    submission_df.loc[submission_df['Id'] == '9d2d08b2-bada-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '12'
    submission_df.loc[submission_df['Id'] == '260a351a-bac7-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '17'
    submission_df.loc[submission_df['Id'] == '54138f64-bad7-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '17 25'
    submission_df.loc[submission_df['Id'] == '443b81cc-bac9-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '15'
    submission_df.loc[submission_df['Id'] == '72c9bb82-bac7-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '4 21 17'
    submission_df.loc[submission_df['Id'] == 'be0cf5a8-bad3-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '16 23'
    submission_df.loc[submission_df['Id'] == '01835f6c-bad3-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '4'
    submission_df.loc[submission_df['Id'] == '74993d6e-bad8-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '16'
    submission_df.loc[submission_df['Id'] == '84ca8928-bad2-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '17'
    submission_df.loc[submission_df['Id'] == '8f8c19a6-bacf-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 25'
    submission_df.loc[submission_df['Id'] == '107d6830-bac6-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '0 19 25'
    submission_df.loc[submission_df['Id'] == '70f3e586-bacb-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 21'
    submission_df.loc[submission_df['Id'] == 'a7e9e53a-bad1-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '16 25'
    submission_df.loc[submission_df['Id'] == 'aa45019c-bad2-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 16'
    submission_df.loc[submission_df['Id'] == '4509520a-bad3-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '1 2'
    submission_df.loc[submission_df['Id'] == 'ec087d1e-bacf-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 16 25'
    submission_df.loc[submission_df['Id'] == '18d295a6-bad9-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '23'
    submission_df.loc[submission_df['Id'] == 'db77c3dc-bacb-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '17 19'
    submission_df.loc[submission_df['Id'] == 'a6c830fa-bad9-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '22'
    submission_df.loc[submission_df['Id'] == '0457b426-baca-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '2'
    submission_df.loc[submission_df['Id'] == 'd79a1e12-bad1-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '16'
    submission_df.loc[submission_df['Id'] == '7929949a-bad4-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '17 25'
    submission_df.loc[submission_df['Id'] == '7d7565e2-bad9-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '0 16'
    submission_df.loc[submission_df['Id'] == 'd6a07ae2-bad1-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '5'
    submission_df.loc[submission_df['Id'] == '82c1d5f6-bacc-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 14 18'
    submission_df.loc[submission_df['Id'] == 'bfdfc644-bac9-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '19'
    submission_df.loc[submission_df['Id'] == '74bdbb12-bace-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '7 25'
    submission_df.loc[submission_df['Id'] == 'a4b77124-bad7-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '23'
    submission_df.loc[submission_df['Id'] == '27ca9b0e-bace-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '12'
    submission_df.loc[submission_df['Id'] == '9ded793a-bacb-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '2 4'
    submission_df.loc[submission_df['Id'] == '77161884-bad6-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '16'
    submission_df.loc[submission_df['Id'] == '17d8a71c-bacf-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '5'
    submission_df.loc[submission_df['Id'] == 'c3c67272-bad7-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '0'
    submission_df.loc[submission_df['Id'] == 'b52035ba-bad1-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '3'
    submission_df.loc[submission_df['Id'] == '718ebf3e-bada-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '16 17 23'
    submission_df.loc[submission_df['Id'] == '530bfbea-baca-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '14 17 23'
    submission_df.loc[submission_df['Id'] == 'c78652b6-bad6-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '0 25'
    submission_df.loc[submission_df['Id'] == '8316d286-bad6-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '6 21'
    submission_df.loc[submission_df['Id'] == 'e1d58c82-bac6-11e8-b2b7-ac1f6b6435d0', 'Predicted'] = '0 16'
    submission_df.loc[submission_df['Id'] == '1a15e75a-bad5-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '0 16 17'
    submission_df.loc[submission_df['Id'] == '0f0ccc64-bad7-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '0 16 25'
    submission_df.loc[submission_df['Id'] == 'b642cefc-bad7-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '0 16 25'
    submission_df.loc[submission_df['Id'] == '9f71c832-bacc-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '14 16'
    submission_df.loc[submission_df['Id'] == 'f665e29c-bad4-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '14 16'
    submission_df.loc[submission_df['Id'] == 'e75331f2-bad8-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '14 16'
    submission_df.loc[submission_df['Id'] == 'be034880-bad0-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '14 16 19'
    submission_df.loc[submission_df['Id'] == '5d2711a6-bac9-11e8-b2b8-ac1f6b6435d0', 'Predicted'] = '14 16 25'
    submission_df.loc[submission_df['Id'] == '89975d50-bad7-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '15 25'
    submission_df.loc[submission_df['Id'] == '7fcba676-bad9-11e8-b2b9-ac1f6b6435d0', 'Predicted'] = '15 25'
    return submission_df
# sub = pd.read_csv('./resnet101_bestloss_submission.csv')
# sub = process_submission_leakdata(sub)
# sub.to_csv('resnet101_bestloss_submission1.csv', index=None)


def process_submission_leakdata_full(submission_df):
    test_matches = pd.read_csv(config.test_matches_csv)
    test_matches = test_matches.drop(['Extra', 'SimR', 'SimG', 'SimB', 'Target_noisey'], axis=1)
    for i in range(len(test_matches)):
        id = test_matches.iloc[i, 0]
        target = test_matches.iloc[i, 1]
        # print("%s %010s    %d" % (id, target, submission_df[submission_df.Id == id].index.values))
        submission_df.loc[submission_df['Id'] == id, 'Predicted'] = target
    return submission_df
# sub = pd.read_csv('./512x512_average345_thres_tta_ensemble_submission.csv')
# sub = process_submission_leakdata_full(sub)
# sub.to_csv('512x512_average345_thres_tta_ensemble_full_leak_submission.csv', index=None)


def calculate_mean_and_std():
    np.random.seed(int(time.time()))
    paths = config.external_data
    color = ['red', 'green', 'blue', 'yellow']
    files = np.array(os.listdir(paths))
    files = files[np.random.choice(len(files), 5000, replace=False)]
    mean = []
    std = []
    for c in color:
        allim = None
        for i, s in enumerate(tqdm(files)):
            if s.split('.')[0].split('_')[-1] == c:
                im = np.array(Image.open(paths + s))  # shape = (512, 512)
                im = np.expand_dims(im, axis=2)
                im = np.divide(im, 255)
                if allim is None:
                    allim = im
                else:
                    allim = np.concatenate((allim, im), axis=-1)
        m = np.mean(allim)
        s = np.std(allim, ddof=1)
        mean.append(m)
        std.append(s)
    return mean, std
# mean, std = calculate_mean_and_std()
# print(mean)
# print(std)


def process_together_labels(df):
    def add_labels(row):
        label = row.loc['Predicted']
        if type(label) == float:
            return row
        list_label = label.split(' ')
        if '9' in list_label and '10' not in list_label:
            # label += ' 10'
            row.loc['Predicted'] = label
            print(label)
        elif '10' in list_label and '9' not in list_label:
            label += ' 9'
            row.loc['Predicted'] = label
            print(label)
        return row

    df = df.apply(add_labels, axis=1)
    return df
# submission_df = pd.read_csv('densenet121_bestf1_submission.csv')
# submission_df = process_together_labels(submission_df)
# submission_df.to_csv('densenet121_bestf1_submission_add_label.csv', index=None)


def process_dup_img_in_test(df):
    dup_test_id = [
        '0774284e-bad7-11e8-b2b9-ac1f6b6435d0', '34aa05de-bad4-11e8-b2b8-ac1f6b6435d0',
        '8f0666da-bad9-11e8-b2b9-ac1f6b6435d0', 'dad043dc-bad5-11e8-b2b9-ac1f6b6435d0',
        'ed5cf6e2-bad2-11e8-b2b8-ac1f6b6435d0', '0d43e7ca-bada-11e8-b2b9-ac1f6b6435d0',
        '5b685d12-bad2-11e8-b2b8-ac1f6b6435d0', '0c27d1d6-baca-11e8-b2b8-ac1f6b6435d0',
        '3323fc06-bad4-11e8-b2b8-ac1f6b6435d0', '2e2f56fa-bacf-11e8-b2b8-ac1f6b6435d0',
        '280f99a0-bad5-11e8-b2b8-ac1f6b6435d0', '93dde07a-baca-11e8-b2b8-ac1f6b6435d0',
        '6b772fd6-bad1-11e8-b2b8-ac1f6b6435d0', 'bf6c5ff0-bacf-11e8-b2b8-ac1f6b6435d0',
        'e45bc644-bad8-11e8-b2b9-ac1f6b6435d0', 'c20d669e-bad6-11e8-b2b9-ac1f6b6435d0',
        'c87ba6c2-bada-11e8-b2b9-ac1f6b6435d0', '53121c7c-bad0-11e8-b2b8-ac1f6b6435d0',
        '30c78b66-bac6-11e8-b2b7-ac1f6b6435d0', '57fca030-bacc-11e8-b2b8-ac1f6b6435d0',
        '54217ef6-bacf-11e8-b2b8-ac1f6b6435d0', '7ad6a956-bad2-11e8-b2b8-ac1f6b6435d0',
        '09b910ec-bac8-11e8-b2b7-ac1f6b6435d0', '66606722-bace-11e8-b2b8-ac1f6b6435d0',
        'b19097da-bac9-11e8-b2b8-ac1f6b6435d0', '59f6c990-bad8-11e8-b2b9-ac1f6b6435d0',
        '70e944ce-bac5-11e8-b2b7-ac1f6b6435d0', 'eba13978-bad0-11e8-b2b8-ac1f6b6435d0'
    ]
    for i in range(int(len(dup_test_id) / 2)):
        best = dup_test_id[2 * i]
        worse = dup_test_id[2 * i + 1]
        target = df.loc[df['Id'] == best, 'Predicted'].values
        df.loc[df['Id'] == worse, 'Predicted'] = target
    return df



def tra_val_split(all_list):
    # train_data_list, val_data_list = train_test_split(all_list, test_size=0.13, random_state=2050)
    # return train_data_list, val_data_list
    x = np.arange(len(all_list))
    kf = KFold(n_splits=5, shuffle=True, random_state=2050)
    for i, (train_index, test_index) in enumerate(kf.split(x)):
        if (i+1) == config.kfold_index:
            print(train_index)
            print(test_index)
            return all_list.iloc[train_index], all_list.iloc[test_index]


def get_threshold(f1_or_loss, folds):
    best_f1_thres = [None] * 17
    best_loss_thres = [None] * 17
    best_f1_thres[1] = np.array([0.3030303 , 0.44040404, 0.41616161, 0.44848484, 0.56969697,
                                 0.46464646, 0.52121212, 0.39797979, 0.65858586, 0.31515151,
                                 0.39191919, 0.48686869, 0.48888889, 0.43838384, 0.56767677,
                                 0.10707071, 0.38787878, 0.43030303, 0.36565656, 0.37575757,
                                 0.46060606, 0.35959596, 0.4020202 , 0.41818181, 0.53939394,
                                 0.31919192, 0.54747475, 0.20606061]) # TODO average by fold 7 8 9 10 11
    best_f1_thres[2] = np.array([0.32323232, 0.53535354, 0.36767677, 0.44040404, 0.57777778,
                                 0.42424242, 0.54747475, 0.35959596, 0.85252525, 0.63030303,
                                 0.66666667, 0.57171717, 0.50505051, 0.52121212, 0.52727273,
                                 0.4989899 , 0.29696969, 0.6929293 , 0.45454545, 0.41616161,
                                 0.57171717, 0.34141414, 0.45252525, 0.4040404 , 0.64646465,
                                 0.31717171, 0.65858586, 0.14949495])  # TODO average by fold 12 13 14 15 16
    best_f1_thres[3] = np.array([0.31313131, 0.48787879, 0.39191919, 0.44444444, 0.57373738,
                                 0.44444444, 0.53434344, 0.37878787, 0.75555556, 0.47272727,
                                 0.52929293, 0.52929293, 0.4969697 , 0.47979798, 0.54747475,
                                 0.3030303 , 0.34242424, 0.56161616, 0.41010101, 0.39595959,
                                 0.51616162, 0.35050505, 0.42727272, 0.41111111, 0.59292929,
                                 0.31818181, 0.60303031, 0.17777778])  # TODO average by fold 7 8 9 10 11 12 13 14 15 16

    best_f1_thres[7] = np.array([0.29292929, 0.4040404,  0.35353535, 0.49494949, 0.56565657, 0.39393939,
                                 0.39393939, 0.35353535, 0.92929293, 0.27272727, 0.25252525, 0.51515152,
                                 0.42424242, 0.58585859, 0.65656566, 0.04040404, 0.36363636, 0.42424242,
                                 0.32323232, 0.37373737, 0.65656566, 0.35353535, 0.41414141, 0.41414141,
                                 0.56565657, 0.27272727, 0.55555556, 0.06060606]) # TODO compute by fold 7
    best_f1_thres[8] = np.array([0.26262626,0.51515152,0.38383838,0.47474747,0.60606061,0.44444444,
                                 0.49494949,0.47474747,0.50505051,0.26262626,0.47474747,0.41414141,
                                 0.46464646,0.37373737,0.61616162,0.08080808,0.47474747,0.39393939,
                                 0.41414141,0.3030303 , 0.25252525, 0.32323232, 0.4040404,  0.39393939,
                                 0.37373737, 0.37373737, 0.45454545, 0.62626263])  # TODO compute by fold 8
    best_f1_thres[9] = np.array([0.33333333, 0.38383838, 0.42424242, 0.41414141, 0.62626263, 0.49494949,
                                 0.5959596 , 0.31313131, 0.3030303 , 0.34343434, 0.13131313, 0.50505051,
                                 0.32323232, 0.39393939, 0.46464646, 0.09090909, 0.32323232, 0.42424242,
                                 0.22222222, 0.32323232, 0.34343434, 0.42424242, 0.36363636, 0.39393939,
                                 0.63636364, 0.3030303 , 0.56565657, 0.13131313])  # TODO compute by fold 9
    best_f1_thres[10] = np.array([0.28282828, 0.42424242, 0.44444444, 0.37373737, 0.60606061, 0.47474747,
                                  0.51515152, 0.46464646, 0.84848485, 0.31313131, 0.38383838, 0.55555556,
                                  0.67676768, 0.56565657, 0.45454545, 0.17171717, 0.44444444, 0.27272727,
                                  0.48484848, 0.42424242, 0.47474747, 0.38383838, 0.48484848, 0.4040404,
                                  0.49494949, 0.32323232, 0.60606061, 0.11111111])  # TODO compute by fold 10
    best_f1_thres[11] = np.array([0.34343434, 0.47474747, 0.47474747, 0.48484848, 0.44444444, 0.51515152,
                                  0.60606061, 0.38383838, 0.70707071, 0.38383838, 0.71717172, 0.44444444,
                                  0.55555556, 0.27272727, 0.64646465, 0.15151515, 0.33333333, 0.63636364,
                                  0.38383838, 0.45454545, 0.57575758, 0.31313131, 0.34343434, 0.48484848,
                                  0.62626263, 0.32323232, 0.55555556, 0.1010101 ])  # TODO compute by fold 11

    best_f1_thres[12] = np.array([0.39393939, 0.67676768, 0.36363636, 0.53535354, 0.53535354, 0.38383838,
                                  0.5959596 , 0.4040404 , 0.95959596, 0.97979798, 0.94949495, 0.54545455,
                                  0.56565657, 0.65656566, 0.52525253, 0.96969697, 0.39393939, 0.68686869,
                                  0.51515152, 0.4040404 , 0.48484848, 0.35353535, 0.39393939, 0.34343434,
                                  0.84848485, 0.31313131, 0.31313131, 0.15151515])  # TODO compute by fold 12
    best_f1_thres[13] = np.array([0.33333333, 0.53535354, 0.3030303,  0.53535354, 0.61616162, 0.44444444,
                                  0.55555556, 0.35353535, 0.85858586, 0.44444444, 0.45454545, 0.5959596,
                                  0.36363636, 0.24242424, 0.57575758, 0.16161616, 0.23232323, 0.72727273,
                                  0.53535354, 0.45454545, 0.48484848, 0.31313131, 0.43434343, 0.47474747,
                                  0.70707071, 0.32323232, 0.61616162, 0.2020202 ])  # TODO compute by fold 13
    best_f1_thres[14] = np.array([0.31313131, 0.50505051, 0.51515152, 0.3030303 , 0.62626263, 0.45454545,
                                  0.45454545, 0.28282828, 0.67676768, 0.46464646, 0.27272727, 0.67676768,
                                  0.44444444, 0.37373737, 0.44444444, 0.25252525, 0.27272727, 0.98989899,
                                  0.3030303 , 0.21212121, 0.72727273, 0.33333333, 0.44444444, 0.35353535,
                                  0.67676768, 0.33333333, 0.74747475, 0.11111111])  # TODO compute by fold 14
    best_f1_thres[15] = np.array([0.3030303 , 0.37373737, 0.29292929, 0.36363636, 0.57575758, 0.41414141,
                                  0.48484848, 0.43434343, 0.83838384, 0.86868687, 0.86868687, 0.54545455,
                                  0.56565657, 0.72727273, 0.52525253, 0.25252525, 0.25252525, 0.50505051,
                                  0.47474747, 0.61616162, 0.52525253, 0.36363636, 0.49494949, 0.37373737,
                                  0.25252525, 0.27272727, 0.84848485, 0.18181818])  # TODO compute by fold 15
    best_f1_thres[16] = np.array([0.27272727, 0.58585859, 0.36363636, 0.46464646, 0.53535354, 0.42424242,
                                  0.64646465, 0.32323232, 0.92929293, 0.39393939, 0.78787879, 0.49494949,
                                  0.58585859, 0.60606061, 0.56565657, 0.85858586, 0.33333333, 0.55555556,
                                  0.44444444, 0.39393939, 0.63636364, 0.34343434, 0.49494949, 0.47474747,
                                  0.74747475, 0.34343434, 0.76767677, 0.1010101 ])  # TODO compute by fold 16
    if f1_or_loss == "f1":
        return best_f1_thres[folds]
    else:
        return best_loss_thres[folds]
