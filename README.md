# [Human Protein Atlas Image Classification](https://www.kaggle.com/c/human-protein-atlas-image-classification)
## Best solution
|rank   |solution             |github         |author                   |
|-------|---------------------|---------------|-------------------------|
|1st|[1st Place Solution](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109)|Github code|bestfitting|
|3rd|[3rd Place Solution](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77320)|Github code|pudae|
|4th|[4th Place Solution](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77300)|Github code|Dieter|
|5th|[5th Place Solution](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77731)|Github code|lingyundev|
|7th|[7th Place Solution](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77269)|Github code|Guanshuo Xu|
|8th|[8th Place Solution](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77251)|Github code|Sergei Fironov|
|11th|[11st Place Solution](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77282)|[Github code](https://github.com/Gary-Deeplearning/Human_Protein)|Gary, shisu|
|12nd|[12nd Place Solution](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77325)|Github code|Arnau Raventós|
|15th|[15th Place Solution](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77322)|Github code|NguyenThanhNhan|
|25th|[25th Place Solution](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77276)|Github code|Soonhwan Kwon|
|29th|[29th Place Solution](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77299)|Github code|zhangboshen|
|30th|[30th Place Solution](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77285)|Github code|Bac Nguyen|
|33rd|[33rd Place Solution](https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/77256)|[Github code](https://github.com/ildoonet/kaggle-human-protein-atlas-image-classification)|Ildoo Kim|

## Overview
##### Training
- Framework: `PyTorch`
- Model: `Densenet121`
- Data: `kaggle data`, [`external data`](http://www.proteinatlas.org/)
- Augmentation: `horizontalflip, verticalflip, rotate, shear, lighter/darker`
- Normalize: `kaggle data and external data with different mean and std`
- Optimizer: `SGD`
- Loss: `Binary Cross Entropy Loss (no weight)`
- Learning rate: `starts at 0.03 and ends at 0.00003`
- Scheduler: `Multi Step Learning Rates (by my experience)`
- Data imbalance: `OverSampling`
- Image size: `512`
- Batch size: `8`
- Epochs: `24`
- CV: `5-fold`

##### Prediciton
- Threshold: `search the best threshold for each class with valida set`
- TTA number: `4`
- TTA augmentation: `2-horizontalflip x 2-verticalflip`
- Average mean ensemble: `5-flod * 4-TTA * 3-threshold = 60`

##### Result
- Training takes ~35 hours pre single fold on GTX 1070
- Public LB: `0.566`
- Private LB: `0.546 (28th)`


## Observations
##### Work
- Oversample is useful
- External data helps a lot
- 5 folds improve score by 0.02
- TTA helps too
- Put all images into SSD faster than HDD in training.
- Threshold is crucial, and different threshold have a great influence on the LB score.
The threshold I searched by valid set lowered my score at first. So I use the constant threshold (0.15) for a long time.
The searched threshold for each class varies from 0.1 to 0.9 and different, and I cannot find any relationship between rare class and its threshold.
I just want to give up to use searched threshold until I found that using smaller threshold would increase the score (not always).
So I just multiply each searched threshold by a factor(~0.4). That mean I want predicted more target and get high recall.
Although doing so may lower my f1 score, it does improve my public/private LB score.
I think the reason may be that the wrong classification can be eliminated by tta and ensemble, and finally get more #TP and high score.
##### Didn't rowk
- Weighted BCE loss work worse for me, and I have no time to make it work better.
- Ensemble 256x256 with 512x512 lower my score, and I just discarded the result predicted by smaller images.
- Leak data can only improve public LB, not helpful for private LB.
- Complex models work badly, for example resnet152, densenet161. 
##### Not sure
- Adam doesn't work well on my model, it's most likely that I didn't find a suitable learning rate.
- Weighted ensemble may work, but I think it is easy overfitting to public LB .
- Split the tif images(2048x2048) into patches may helpful. I knew it too late, otherwise I will try it.
- May be RGB work better than RGBY

## Usage
- 1.clone the repository
```
git clone https://github.com/feifei9099/kaggle_human_protein.git
cd kaggle_human_protein
```

- 2.install requirements
```
conda create --name kaggle python=3.6
source activate kaggle
pip install numpy==1.15.4 torch==0.4.0 torchvision==0.2.1 scikit-learn==0.20.0 pandas==0.23.4 imgaug==0.2.6 tqdm==4.29.1 pretrainedmodels==0.7.4
conda install -c menpo opencv3 
```

- 3.download data
```
kaggle competitions download -c human-protein-atlas-image-classification
python my_utils/download.py
```

- 4.update `config.py` file to match your preferences
```python
train_data = "path_to_your_train_data/train/"
test_data = "path_to_your_train_data/test/"
external_data = "path_to_your_train_data/external_data_HPAv18/"
test_csv = "path_to_your_sub_csv/sample_submission.csv"
train_csv = "path_to_your_train_csv/train.csv"
external_csv = "path_to_your_external_csv/external_data_HPAv18.csv"
```

- 5.train you model 5 times (cv)
```
python main.py
```

- 6.update `config.py` file and rerun main.py to predict
```python
is_train = False
is_test = True
```

- 7.ensemble submission file
```
python my_utils/kfold_cross_validation.py
```


## Code Interpretation
- Properly process external data is key to improve scores.
The red, green and blue images directly extract the corresponding channels of original jpg images and save them into 512x512 gray png images.
The yellow image combine R and Y channel in original image.
```python
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
```

- kaggle set and external set use two different mean and std, which can be calculated with the following code.
```python
T.Normalize([0.0789, 0.0529, 0.0546, 0.0814], [0.147, 0.113, 0.157, 0.148]) # kaggle set
T.Normalize([0.1177, 0.0696, 0.0660, 0.1056], [0.179, 0.127, 0.176, 0.166]) # external set
```
```python
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
```

- the oversampling weight is calculated by #class_target/#total_target or math.log(#class_target/#total_target). I use the log weight and put it in DataFrame's freq column
```python
sampler = WeightedRandomSampler(train_data_list['freq'].values, num_samples=int(len(train_data_list)*config.multiply), replacement=True)
train_loader = DataLoader(train_gen,batch_size=config.batch_size,drop_last=True,sampler=sampler,pin_memory=True,num_workers=6)
```

- average mean ensemble code shown as follows, which boost my score 0.02
```python
for i, file in enumerate(sub_files):
    file_path = sub_path + file
    df = pd.read_csv(file_path)
    df = df.fillna('28')
    # df['pre_vec'] = df['Predicted'].map(lambda x: list(map(int, x.strip().split())))
    sub[i] = df
for p in range(len(sample_submission_df)):
    all_target = np.zeros((1, 28))
    for s in range(lg):
        row1 = sub[s].iloc[p]
        target = row1.Predicted
        target = list(map(int, target.strip().split()))
        target_array = np.zeros((1, 28))
        for n in target:
            if n == 28:
                continue
            target_array[:, n] = 1
        all_target += target_array
    all_target = all_target / lg > 0.5
    labels.append(all_target)
```

- search threshold for each class
```python
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
```


## For help
- If someone have any questions and suggestions,please tell me!!!<br>
Email: scsncfb@126.com




