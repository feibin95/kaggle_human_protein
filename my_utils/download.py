import os
from multiprocessing.pool import Pool
from tqdm import tqdm
import requests
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# def download(pid, sp, ep):
#     colors = ['red', 'green', 'blue', 'yellow']
#     image_size = (512, 512)
#     DIR = "/media/femi/Data/datasets/HumanProtein/external_data_HPAv18/"
#     v18_url = 'http://v18.proteinatlas.org/images/'
#     imgList = pd.read_csv("/media/femi/Data/datasets/HumanProtein/external_data_HPAv18.csv")
#     for i in tqdm(imgList['Id'][sp:ep], postfix=pid):  # [:5] means downloard only first 5 samples
#         img = i.split('_')
#         for color in colors:
#             img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
#             img_name = i + "_" + color + ".png"
#             img_url = v18_url + img_path
#             r = requests.get(img_url, allow_redirects=True, stream=True)
#             r.raw.decode_content = True;
#             image = Image.open(r.raw).resize(image_size, Image.LANCZOS).convert('L')
#             image.save(DIR + img_name)

# def download(pid, sp, ep):
#     colors = ['red', 'green', 'blue', 'yellow']
#     image_size = (512, 512)
#     DIR = "/media/femi/Data/datasets/HumanProtein/external_data_HPAv18/"
#     v18_url = 'http://v18.proteinatlas.org/images/'
#     imgList = pd.read_csv("/media/femi/Data/datasets/HumanProtein/external_data_HPAv18.csv")
#     for i in tqdm(imgList['Id'][sp:ep], postfix=pid):  # [:5] means downloard only first 5 samples
#         img = i.split('_')
#         for color in colors:
#             img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
#             img_name = i + "_" + color + ".png"
#             img_url = v18_url + img_path
#             r = requests.get(img_url, allow_redirects=True, stream=True)
#             r.raw.decode_content = True
#             im = Image.open(r.raw)
#             r, g, b = im.resize(image_size, Image.LANCZOS).split()
#             if color == 'red':
#                 im = r
#             elif color == 'green':
#                 im = g
#             elif color == 'blue':
#                 im = b
#             else:
#                 im = Image.blend(r, g, 0.5)
#             im.save(os.path.join(DIR, img_name), 'PNG')


files = os.listdir('/media/femi/Data/datasets/HumanProtein/external_data_HPAv18')
csv_file = "./my_utils/external_data_HPAv18.csv"
# csv_file = "add_file.csv"

def get_html(url):
    i = 0
    while i < 3:
        try:
            r = requests.get(url, timeout=100)
            return r
        except requests.exceptions.RequestException:
            i += 1
    open('log', 'a').write(url)
    return -1


def download(pid, sp, ep):
    colors = ['red', 'green', 'blue', 'yellow']
    image_size = (512, 512)
    DIR = "/media/femi/Data/datasets/HumanProtein/external_data_HPAv18/"
    v18_url = 'http://v18.proteinatlas.org/images/'
    imgList = pd.read_csv(csv_file)
    for i in tqdm(imgList['Id'][sp:ep], postfix=pid):
        img = i.split('_')
        for color in colors:
            img_path = img[0] + '/' + "_".join(img[1:]) + "_" + color + ".jpg"
            img_name = i + "_" + color + ".jpg"
            img_name2 = i + "_" + color + ".png"
            img_url = v18_url + img_path
            if img_name2 in files:
                continue
            r = get_html(img_url)
            if isinstance(r, int) and r == -1:
                continue
            open(DIR + img_name, 'wb').write(r.content)
            im = Image.open(DIR + img_name)
            os.remove(DIR + img_name)
            r, g, b = im.resize(image_size, Image.LANCZOS).split()
            if color == 'red':
                im = r
            elif color == 'green':
                im = g
            elif color == 'blue':
                im = b
            else:
                im = Image.blend(r, g, 0.5)
            im.save(DIR + img_name2, 'PNG')
            print('save')
            break


def run_proc(name, sp, ep):
    print('Run child process %s (%s) sp:%d ep: %d' % (name, os.getpid(), sp, ep))
    download(name, sp, ep)
    print('Run child process %s done' % (name))


if __name__ == "__main__":
    print('Parent process %s.' % os.getpid())
    img_list = pd.read_csv(csv_file)['Id']
    list_len = len(img_list)
    process_num = 100
    open('log', 'a').write('\n')
    p = Pool(process_num)
    for i in range(process_num):
        p.apply_async(run_proc, args=(str(i), int(i * list_len / process_num), int((i + 1) * list_len / process_num)))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')



