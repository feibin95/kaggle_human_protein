import os
import pandas as pd
from tqdm import tqdm

files = os.listdir('/media/femi/Data/datasets/HumanProtein/external_data_HPAv18')
DIR = "/media/femi/Data/datasets/HumanProtein/external_data_HPAv18/"
img_list = pd.read_csv("/media/femi/Data/datasets/HumanProtein/external_data_HPAv18.csv")['Id']
colors = ['red', 'green', 'blue', 'yellow']
count = 0
add = []
for img in tqdm(img_list):
    flag = True
    for c in colors:
        img_name = img + '_' + c + '.png'
        if img_name not in files:
            flag = False
            count += 1
            print(img_name)
            open('check', 'a').write(img_name + '\n')
    if not flag:
        add.append(img)
if add != []:
    df = pd.DataFrame({'Id': add})
    df.to_csv('add_file.csv', index=False)
print(count)