import os
import time
import json
import random
import warnings
import numpy as np
import pandas as pd
import pathlib
import termios

from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import f1_score
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain

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

dup_id = [
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

test_data = "/home/femi/work/datasets/HumanProtein/test/"
test_csv = "/home/femi/work/datasets/HumanProtein/sample_submission.csv"
train_data = "/home/femi/work/datasets/HumanProtein/train/"
train_csv = "/home/femi/work/datasets/HumanProtein/train.csv"
external_data = "/home/femi/work/datasets/HumanProtein/external_data_HPAv18/"
external_csv = "/home/femi/work/datasets/HumanProtein/external_data_HPAv18.csv"


def view_all_dup_images():
    for i in range(int(len(dup_id) / 2)):
        name1 = dup_id[2 * i]
        name2 = dup_id[2 * i + 1]
        path1 = test_data + name1
        path2 = test_data + name2
        r1 = Image.open(path1 + "_red.png")
        g1 = Image.open(path1 + "_green.png")
        b1 = Image.open(path1 + "_blue.png")
        y1 = Image.open(path1 + "_yellow.png")
        r2 = Image.open(path2 + "_red.png")
        g2 = Image.open(path2 + "_green.png")
        b2 = Image.open(path2 + "_blue.png")
        y2 = Image.open(path2 + "_yellow.png")
        images1 = np.zeros(shape=(512, 512, 3), dtype=np.float)
        images2 = np.zeros(shape=(512, 512, 3), dtype=np.float)
        images1[:, :, 0] = np.array(r1) / 2 + np.array(y1) / 2
        images1[:, :, 1] = g1
        images1[:, :, 2] = b1
        images1 = images1.astype(np.uint8)
        images2[:, :, 0] = np.array(r2) / 2 + np.array(y2) / 2
        images2[:, :, 1] = g2
        images2[:, :, 2] = b2
        images2 = images2.astype(np.uint8)
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(images1)
        ax[0].set_title(name1)
        ax[1].imshow(images2)
        ax[1].set_title(name2)
        plt.waitforbuttonpress(0)
        plt.close(fig)
# view_all_dup_images()


def view_one_image():
    name = '3aee7906-bba2-11e8-b2b9-ac1f6b6435d0'
    path = train_data + name
    r = Image.open(path + "_red.png")
    g = Image.open(path + "_green.png")
    b = Image.open(path + "_blue.png")
    y = Image.open(path + "_yellow.png")
    images = np.zeros(shape=(512, 512, 3), dtype=np.float)
    images[:, :, 0] = np.array(r) / 2 + np.array(y) / 2
    images[:, :, 1] = g
    images[:, :, 2] = b
    images = images.astype(np.uint8)
    fig, ax = plt.subplots(1, 5, figsize=(25, 5))
    # ax[0].imshow((np.array(r) * 2).astype(np.uint8))
    # ax[1].imshow((np.array(r) * 2).astype(np.uint8))
    # ax[2].imshow((np.array(r) * 2).astype(np.uint8))
    # ax[3].imshow((np.array(r) * 2).astype(np.uint8))
    # ax[4].imshow(images * 10)
    ax[0].imshow(r)
    ax[1].imshow(g)
    ax[2].imshow(b)
    ax[3].imshow(y)
    ax[4].imshow(images)
    plt.show()
view_one_image()


def view_all_test_Nan(df):
    nan_df = df.loc[pd.isna(df['Predicted'])]
    for i in range(int(len(nan_df) / 2)):
        name1 = nan_df.iloc[2 * i].Id
        name2 = nan_df.iloc[2 * i + 1].Id
        path1 = test_data + name1
        path2 = test_data + name2
        r1 = Image.open(path1 + "_red.png")
        g1 = Image.open(path1 + "_green.png")
        b1 = Image.open(path1 + "_blue.png")
        y1 = Image.open(path1 + "_yellow.png")
        r2 = Image.open(path2 + "_red.png")
        g2 = Image.open(path2 + "_green.png")
        b2 = Image.open(path2 + "_blue.png")
        y2 = Image.open(path2 + "_yellow.png")
        images1 = np.zeros(shape=(512, 512, 3), dtype=np.float)
        images2 = np.zeros(shape=(512, 512, 3), dtype=np.float)
        images1[:, :, 0] = np.array(r1) / 2 + np.array(y1) / 2
        images1[:, :, 1] = g1
        images1[:, :, 2] = b1
        images1 = images1.astype(np.uint8)
        images2[:, :, 0] = np.array(r2) / 2 + np.array(y2) / 2
        images2[:, :, 1] = g2
        images2[:, :, 2] = b2
        images2 = images2.astype(np.uint8)
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(images1)
        ax[0].set_title(name1 + '    ' + str(2 * i))
        ax[1].imshow(images2)
        ax[1].set_title(name2)
        plt.waitforbuttonpress(0)
        plt.close(fig)
# test_df = pd.read_csv("/home/femi/work/study/kaggle/protein/pytorch_densenet121_externaldata/submit/densenet121_best_f1_submission.csv")
# view_all_test_Nan(test_df)


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
    for i in range(int(len(dup_id) / 2)):
        best = dup_test_id[2 * i]
        worse = dup_test_id[2 * i + 1]
        target = df.loc[df['Id'] == best, 'Predicted']
        df.loc[df['Id'] == worse, 'Predicted'] = target
    return df


def view_all_img(datapath, df):
    row = 4
    col = 5
    for i in range(int(len(df) / row / col)):
        fig, ax = plt.subplots(row, col, figsize=(row * 5, col * 5))
        for j in range(row):
            for k in range(col):
                name = df.iloc[row * col * i + col * j + k].Id
                path = datapath + name
                r = Image.open(path + "_red.png")
                g = Image.open(path + "_green.png")
                b = Image.open(path + "_blue.png")
                y = Image.open(path + "_yellow.png")
                images = np.zeros(shape=(512, 512, 3), dtype=np.float)
                images[:, :, 0] = np.array(r) / 2 + np.array(y) / 2
                images[:, :, 1] = g
                images[:, :, 2] = b
                images = images.astype(np.uint8)
                ax[j, k].imshow(images)
                ax[j, k].set_title(name[:20])
        plt.waitforbuttonpress(0)
        plt.close(fig)
# df = pd.read_csv(external_csv)
# view_all_img(external_data, df)


def view_only_one_class(datapath, df):
    # 0: 12885,
    # 25: 8228,
    # 21: 3777,
    # 2: 3621,
    # 23: 2965,
    # 7: 2822,
    # 5: 2513,
    # 4: 1858,
    # 3: 1561,
    # 19: 1482,
    # 1: 1254,
    # 11: 1093,
    # 14: 1066,
    # 6: 1008,
    # 18: 902,
    # 22: 802,
    # 12: 688,
    # 13: 537,
    # 16: 530,
    # 26: 328,
    # 24: 322,
    # 17: 210,
    # 20: 172,
    # 8: 53,
    # 9: 45,
    # 10: 28,
    # 15: 21,
    # 27: 11
    class_num = 24
    df['target_vec'] = df['Target'].map(lambda x: list(map(int, x.strip().split())))
    def chose_or_not(row):
        if class_num in row['target_vec']:
            row['chose'] = 1
        else:
            row['chose'] = 0
        return row
    df = df.apply(chose_or_not, axis=1)
    df = df.loc[df['chose'] == 1]
    print(df)
    row = 4
    col = 5
    for i in range(int(len(df) / row / col + 1)):
        fig, ax = plt.subplots(row, col, figsize=(row * 5, col * 5))
        for j in range(row):
            for k in range(col):
                if (row * col * i + col * j + k) < len(df):
                    name = df.iloc[row * col * i + col * j + k].Id
                    path = datapath + name
                    r = Image.open(path + "_red.png")
                    g = Image.open(path + "_green.png")
                    b = Image.open(path + "_blue.png")
                    y = Image.open(path + "_yellow.png")
                    images = np.zeros(shape=(512, 512, 3), dtype=np.float)
                    images[:, :, 0] = np.array(r) / 2 + np.array(y) / 2
                    images[:, :, 1] = g
                    images[:, :, 2] = b
                    images = images.astype(np.uint8)
                    ax[j, k].imshow(images)
                    ax[j, k].set_title(name[:20])
        plt.waitforbuttonpress(0)
        plt.close(fig)
# df = pd.read_csv(train_csv)
# view_only_one_class(train_data, df)


def view_target_in_sub(df1):
    df1 = df1.loc[~pd.isna(df1['Predicted'])]
    df1['target_vec'] = df1['Predicted'].map(lambda x: list(map(int, x.strip().split())))
    df1 = df1.loc[df1['target_vec'].map(lambda x: 27 in x)]
    print(df1)
# files = "/home/femi/work/study/kaggle/protein/pytorch_densenet121_externaldata/submit/densenet121_best_f1_submission.csv"
# df = pd.read_csv(files)
# view_target_in_sub(df)