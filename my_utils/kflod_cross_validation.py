from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# train_csv = "/home/femi/work/datasets/HumanProtein/train.csv"
# df = pd.read_csv(train_csv)
# x = np.arange(len(df))
# kf = KFold(n_splits=5, shuffle=True)
# for train_index, test_index in kf.split(x):
#     print(df.iloc[train_index])
#     print(df.iloc[test_index])
#     print("\n")



#
# def tra_val_split(all_list):
#     # train_data_list, val_data_list = train_test_split(all_list, test_size=0.13, random_state=2050)
#     # return train_data_list, val_data_list
#     x = np.arange(len(df))
#     kf = KFold(n_splits=5, shuffle=True, random_state=2050)
#     for i, (train_index, test_index) in enumerate(kf.split(x)):
#         if (i+1) == config.kfold_index:
#             return all_list.iloc[train_index], all_list.iloc[test_index]
#
#
#
# def ensemble_test(test_loader,model,folds):
#     sample_submission_df = pd.read_csv(config.test_csv)
#     all_submission, final_submission, str_submission = [], [], []
#     for f in range(5):
#         best_model = torch.load("%s/%s_fold_%s_model_best_%s.pth.tar"%(config.best_models,config.model_name,str(fold + f),config.best))
#         #best_model = torch.load("checkpoints/bninception_bcelog/0/checkpoint.pth.tar")
#         model.load_state_dict(best_model["state_dict"])
#         #3.1 confirm the model converted to cuda
#         labels, submissions= [],[]
#         model.cuda()
#         model.eval()
#         for i,(input,filepath) in enumerate(tqdm(test_loader)):
#             with torch.no_grad():
#                 image_var = input.cuda(non_blocking=True)
#                 y_pred = model(image_var)
#                 label = y_pred.sigmoid().cpu().data.numpy()
#                 threshold = config.threshold
#                 # ll = label.copy().reshape((-1))
#                 # ll = -ll
#                 # ll.sort()
#                 # ll = -ll
#                 # if threshold < ll[3]:
#                 #     threshold = ll[3]
#                 # if threshold > ll[0]:
#                 #     threshold = ll[0]
#                 labels.append(label >= threshold)
#             submissions = np.concatenate(labels)
#         all_submission.append(submissions)
#
#     for i in range(len(sample_submission_df)):
#         sub1 = all_submission[0][i]
#         sub2 = all_submission[1][i]
#         sub3 = all_submission[2][i]
#         sub4 = all_submission[3][i]
#         sub5 = all_submission[4][i]
#         sub = [(sub1[i]+sub2[i]+sub3[i]+sub4[i]+sub5[i])/5 > 0.5 for i in range(len(sub1))]
#         final_submission.append(sub)
#     for row in final_submission:
#         subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
#         str_submission.append(subrow)
#     sample_submission_df['Predicted'] = str_submission
#     sample_submission_df = process_submission_leakdata_full(sample_submission_df)
#     # sample_submission_df = process_together_labels(sample_submission_df)
#     sample_submission_df = process_dup_img_in_test(sample_submission_df)
#     sample_submission_df.to_csv('./submit/%s_best_%s_submission.csv'%(config.model_name, config.best), index=None)


def ensemble():
    sub_path = '/home/femi/work/study/kaggle/protein/pytorch_densenet121_externaldata/submit/'

    sub_files = ['densenet121_best_f1_3_fold_12_submission0.csv',
                 'densenet121_best_f1_3_fold_12_submission1.csv',
                 'densenet121_best_f1_3_fold_12_submission2.csv',
                 'densenet121_best_f1_3_fold_12_submission3.csv',
                 'densenet121_best_f1_3_fold_13_submission0.csv',
                 'densenet121_best_f1_3_fold_13_submission1.csv',
                 'densenet121_best_f1_3_fold_13_submission2.csv',
                 'densenet121_best_f1_3_fold_13_submission3.csv',
                 'densenet121_best_f1_3_fold_14_submission0.csv',
                 'densenet121_best_f1_3_fold_14_submission1.csv',
                 'densenet121_best_f1_3_fold_14_submission2.csv',
                 'densenet121_best_f1_3_fold_14_submission3.csv',
                 'densenet121_best_f1_3_fold_15_submission0.csv',
                 'densenet121_best_f1_3_fold_15_submission1.csv',
                 'densenet121_best_f1_3_fold_15_submission2.csv',
                 'densenet121_best_f1_3_fold_15_submission3.csv',
                 'densenet121_best_f1_3_fold_16_submission0.csv',
                 'densenet121_best_f1_3_fold_16_submission1.csv',
                 'densenet121_best_f1_3_fold_16_submission2.csv',
                 'densenet121_best_f1_3_fold_16_submission3.csv',
                 'densenet121_best_f1_4_fold_12_submission0.csv',
                 'densenet121_best_f1_4_fold_12_submission1.csv',
                 'densenet121_best_f1_4_fold_12_submission2.csv',
                 'densenet121_best_f1_4_fold_12_submission3.csv',
                 'densenet121_best_f1_4_fold_13_submission0.csv',
                 'densenet121_best_f1_4_fold_13_submission1.csv',
                 'densenet121_best_f1_4_fold_13_submission2.csv',
                 'densenet121_best_f1_4_fold_13_submission3.csv',
                 'densenet121_best_f1_4_fold_14_submission0.csv',
                 'densenet121_best_f1_4_fold_14_submission1.csv',
                 'densenet121_best_f1_4_fold_14_submission2.csv',
                 'densenet121_best_f1_4_fold_14_submission3.csv',
                 'densenet121_best_f1_4_fold_15_submission0.csv',
                 'densenet121_best_f1_4_fold_15_submission1.csv',
                 'densenet121_best_f1_4_fold_15_submission2.csv',
                 'densenet121_best_f1_4_fold_15_submission3.csv',
                 'densenet121_best_f1_4_fold_16_submission0.csv',
                 'densenet121_best_f1_4_fold_16_submission1.csv',
                 'densenet121_best_f1_4_fold_16_submission2.csv',
                 'densenet121_best_f1_4_fold_16_submission3.csv',
                 'densenet121_best_f1_5_fold_12_submission0.csv',
                 'densenet121_best_f1_5_fold_12_submission1.csv',
                 'densenet121_best_f1_5_fold_12_submission2.csv',
                 'densenet121_best_f1_5_fold_12_submission3.csv',
                 'densenet121_best_f1_5_fold_13_submission0.csv',
                 'densenet121_best_f1_5_fold_13_submission1.csv',
                 'densenet121_best_f1_5_fold_13_submission2.csv',
                 'densenet121_best_f1_5_fold_13_submission3.csv',
                 'densenet121_best_f1_5_fold_14_submission0.csv',
                 'densenet121_best_f1_5_fold_14_submission1.csv',
                 'densenet121_best_f1_5_fold_14_submission2.csv',
                 'densenet121_best_f1_5_fold_14_submission3.csv',
                 'densenet121_best_f1_5_fold_15_submission0.csv',
                 'densenet121_best_f1_5_fold_15_submission1.csv',
                 'densenet121_best_f1_5_fold_15_submission2.csv',
                 'densenet121_best_f1_5_fold_15_submission3.csv',
                 'densenet121_best_f1_5_fold_16_submission0.csv',
                 'densenet121_best_f1_5_fold_16_submission1.csv',
                 'densenet121_best_f1_5_fold_16_submission2.csv',
                 'densenet121_best_f1_5_fold_16_submission3.csv',
                 'densenet121_best_f1_6_fold_12_submission0.csv',
                 'densenet121_best_f1_6_fold_12_submission1.csv',
                 'densenet121_best_f1_6_fold_12_submission2.csv',
                 'densenet121_best_f1_6_fold_12_submission3.csv',
                 'densenet121_best_f1_6_fold_13_submission0.csv',
                 'densenet121_best_f1_6_fold_13_submission1.csv',
                 'densenet121_best_f1_6_fold_13_submission2.csv',
                 'densenet121_best_f1_6_fold_13_submission3.csv',
                 'densenet121_best_f1_6_fold_14_submission0.csv',
                 'densenet121_best_f1_6_fold_14_submission1.csv',
                 'densenet121_best_f1_6_fold_14_submission2.csv',
                 'densenet121_best_f1_6_fold_14_submission3.csv',
                 'densenet121_best_f1_6_fold_15_submission0.csv',
                 'densenet121_best_f1_6_fold_15_submission1.csv',
                 'densenet121_best_f1_6_fold_15_submission2.csv',
                 'densenet121_best_f1_6_fold_15_submission3.csv',
                 'densenet121_best_f1_6_fold_16_submission0.csv',
                 'densenet121_best_f1_6_fold_16_submission1.csv',
                 'densenet121_best_f1_6_fold_16_submission2.csv',
                 'densenet121_best_f1_6_fold_16_submission3.csv'
                 ]
    test_csv = "/home/femi/work/datasets/HumanProtein/sample_submission.csv"
    sample_submission_df = pd.read_csv(test_csv)
    sub_weight = []
    place_weight = {}
    labels = []
    submissions = []
    lg = len(sub_files)
    sub = [None] * lg

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
    for row in np.concatenate(labels):
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    sample_submission_df['Predicted'] = submissions
    print(sample_submission_df)
    sample_submission_df.to_csv('512x512_average3456_thres_tta_no_process_ensemble_submission.csv', index=None)
ensemble()
