import os
import numpy as np
from PIL import Image
from tqdm import tqdm

DIR = "/home/femi/work/datasets/HumanProtein/external_data_HPAv18/"
files = os.listdir(DIR)
count = 0
for s in tqdm(files):
    try:
        im = np.array(Image.open(DIR + s))
        if im.max() == 0 or im.min() == 255:
            open('check', 'a').write(s + '\n')
            count += 1
    except Exception as msg:
        open('wrong', 'a').write(s + '\n')
print(count)
