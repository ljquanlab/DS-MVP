import argparse
import random
import time
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.functional import F
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score, roc_auc_score,roc_curve
import xgboost as xgb



#################################################################################### 测试 模型文件读取
from models import * 
# Best_Model_Path = '../opt_model/20240731_084324.pth'


#################################################################################### 环境参数设置
Random_Seed = 1024
print(Random_Seed)

# Run_Mode = 'all'
# Run_Mode = 'Train'
# Run_Mode = 'Test'

TEST_LST = ['RareD', 'HardD', 'CSPD', 'NSPD']
#################################################################################### 训练 模型文件保存
IS_SAVE = True # 保存
IS_SAVE_PROB = True


# 获取当前时间
now = datetime.now()
formatted_time = now.strftime('%Y%m%d_%H%M%S')
print(formatted_time)
TRAIN_TIME = formatted_time


Model_PATH = '../opt_model/' + str(TRAIN_TIME) +'.pth'


#################################################################################### 超参设置
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

Epoch = 300
BATCH_SIZE = 256
LR = 5e-4
W_D = 1e-3
# LR_STEP = 15
thres_value = 0.5
PATIENCE = 20



####################################################################################
# 设置随机数种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

    



def load_dataset(isTest=False, test_data='RareD'):
    # load data

    if isTest == False:
        feats_path_list = [
            ################# DNA embed
            '../../../feature/3mer_embed/train_ref_3mer_embed.npy',
            '../../../feature/3mer_embed/train_alt_3mer_embed.npy',
            ################# scoresAF
            '../../../feature/genetic/train_conservation_filled.npy',
            '../../../feature/genetic/train_AF_filled.npy',
            ################# DNAShape
            '../../../feature/shape/train_ref_seq_bp9_MGW.npy',
            '../../../feature/shape/train_alt_seq_bp9_MGW.npy',
            ################# gene 
            '../../../feature/gene_desp/train_desp.npy',
            ################# aa
            '../../../feature/aa_onehot/train_ref_aa.npy',
            '../../../feature/aa_onehot/train_alt_aa.npy',

        ]
        label_path = '../../../feature/label/train_label.npy'

    else:
        # test
        if test_data == 'RareD':
            feats_path_list = [
                ################# DNA embed
                '../../../feature/3mer_embed/RareD_ref_3mer_embed.npy',
                '../../../feature/3mer_embed/RareD_alt_3mer_embed.npy',
                ################# scoresAF
                '../../../feature/genetic/RareD_conservation_filled.npy',
                '../../../feature/genetic/RareD_AF_filled.npy',
                ################# DNAShape
                '../../../feature/shape/RareD_ref_seq_bp9_MGW.npy',
                '../../../feature/shape/RareD_alt_seq_bp9_MGW.npy',
                ################# gene 
                '../../../feature/gene_desp/RareD_desp.npy',
                ################# aa
                '../../../feature/aa_onehot/RareD_ref_aa.npy',
                '../../../feature/aa_onehot/RareD_alt_aa.npy',
                ]
            label_path = '../../../feature/label/RareD_label.npy'

        elif test_data == 'HardD':
            feats_path_list = [
                ################# DNA embed
                '../../../feature/3mer_embed/HardD_ref_3mer_embed.npy',
                '../../../feature/3mer_embed/HardD_alt_3mer_embed.npy',
                ################# scoresAF
                '../../../feature/genetic/HardD_conservation_filled.npy',
                '../../../feature/genetic/HardD_AF_filled.npy',
                ################# DNAShape
                '../../../feature/shape/HardD_ref_seq_bp9_MGW.npy',
                '../../../feature/shape/HardD_alt_seq_bp9_MGW.npy',
                ################# gene 
                '../../../feature/gene_desp/HardD_desp.npy',
                ################# aa
                '../../../feature/aa_onehot/HardD_ref_aa.npy',
                '../../../feature/aa_onehot/HardD_alt_aa.npy',

            ]
            label_path = '../../../feature/label/HardD_label.npy'

        elif test_data == 'CSPD':
            feats_path_list = [
                ################# DNA embed
                '../../../feature/3mer_embed/CSPD_ref_3mer_embed.npy',
                '../../../feature/3mer_embed/CSPD_alt_3mer_embed.npy',
                ################# scoresAF
                '../../../feature/genetic/CSPD_conservation_filled.npy',
                '../../../feature/genetic/CSPD_AF_filled.npy',
                ################# DNAShape
                '../../../feature/shape/CSPD_ref_seq_bp9_MGW.npy',
                '../../../feature/shape/CSPD_alt_seq_bp9_MGW.npy',
                ################# gene 
                '../../../feature/gene_desp/CSPD_desp.npy',
                ################# aa
                '../../../feature/aa_onehot/CSPD_ref_aa.npy',
                '../../../feature/aa_onehot/CSPD_alt_aa.npy',

            ]
            label_path = '../../../feature/label/CSPD_label.npy'


        elif test_data == 'NSPD':
            feats_path_list = [
                ################# DNA embed
                '../../../feature/3mer_embed/NSPD_ref_3mer_embed.npy',
                '../../../feature/3mer_embed/NSPD_alt_3mer_embed.npy',
                ################# scoresAF
                '../../../feature/genetic/NSPD_conservation_filled.npy',
                '../../../feature/genetic/NSPD_AF_filled.npy',
                ################# DNAShape
                '../../../feature/shape/NSPD_ref_seq_bp9_MGW.npy',
                '../../../feature/shape/NSPD_alt_seq_bp9_MGW.npy',
                ################# gene 
                '../../../feature/gene_desp/NSPD_desp.npy',
                ################# aa
                '../../../feature/aa_onehot/NSPD_ref_aa.npy',
                '../../../feature/aa_onehot/NSPD_alt_aa.npy',

            ]
            label_path = '../../../feature/label/NSPD_label.npy'


    data_list = []
    for i in range(len(feats_path_list)):
        data = np.load(feats_path_list[i])
        data = torch.Tensor(np.array(data))
        data_list.append(data)

    label = np.load(label_path)
    label =  torch.Tensor(np.array(label))
    return data_list, label



def load_dataloader(data_type='Test', test_data='RareD'):

    train_index_path = '../data_loader/train_index.txt'
    val_index_path = '../data_loader/val_index.txt'


    #(2)加载索引
    if data_type == 'Train':
        with open(train_index_path, 'r') as f:
            train_index = [int(line.strip()) for line in f]
            train_index = torch.LongTensor(train_index)
        data_loader = DataLoader(dataset=train_index, batch_size=BATCH_SIZE, shuffle=True)


    elif data_type == 'Val':
        with open(val_index_path, 'r') as f:
            val_index = [int(line.strip()) for line in f]
            val_index = torch.LongTensor(val_index)
        data_loader = DataLoader(dataset=val_index, batch_size=BATCH_SIZE, shuffle=False)


    else:
        if test_data == 'RareD':
            test_index = list(range(11540))
        elif test_data == 'HardD':
            test_index = list(range(10400))
        elif test_data == 'CSPD':
            test_index = list(range(2161))
        elif test_data == 'NSPD':
            test_index = list(range(891))

        test_index = torch.LongTensor(test_index)
        data_loader = DataLoader(dataset=test_index, batch_size=BATCH_SIZE, shuffle=False)

    return data_loader



def get_feats(data_feats_list, data_labels, index):
    data_lst = []
    for feat in data_feats_list:
        data_lst.append(feat[index].to(device))

    # label
    out_label = data_labels[index].to(device)
    return data_lst, out_label






def main(run_mode='all'):
    setup_seed(Random_Seed)
    
    parser = argparse.ArgumentParser(description='Train or test for binary classification.')

    # 添加位置参数
    parser.add_argument('--run_mode', type=str, choices=['all', 'train', 'test'], default='all', help='Run mode: all, train or test')
    parser.add_argument('--opt_model_path', type=str,  help='Model file path')

    # 解析命令行参数
    args = parser.parse_args()

    # 根据 run_mode 的值来判断 config_file 是否必须提供
    if args.run_mode == 'test' and args.opt_model_path is None:
        parser.error('The following arguments are required when run_mode is "test": opt_model_path')
    
    try:
        # 根据 run_mode 执行不同的操作
    

        model_mut = DS_MVP_Model().to(device)
        print(model_mut)
        train_lst, train_labels = load_dataset()
        print('* ' * 50)
        
        
        if args.run_mode == 'train':
            print('start train...')
            print('train Model_PATH:', Model_PATH)
            train_func(train_lst, train_labels, model_mut)


        elif args.run_mode == 'test':
            print('start test...')

            best_model_path = args.opt_model_path
            model_mut.load_state_dict(torch.load(best_model_path)) #加载

            test_func(train_lst, train_labels, model_mut)

        else:
            print('start train and test...')

            best_mut_model  = train_func(train_lst, train_labels, model_mut)
            test_func(train_lst, train_labels, best_mut_model)

    except FileNotFoundError:
            print(f"Error: File not found '{args.opt_model_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

################################################# DL
def train_func(train_lst, train_labels, model_mut):

    cri_ce = nn.BCELoss().to(device)
    train_loader = load_dataloader('Train')
    val_loader = load_dataloader('Val')

    optimizer = optim.Adam(model_mut.parameters(), lr=LR, weight_decay=W_D)

    # earlystop
    cur_wait = 0
    best_val_loss = np.inf
    best_mut_model = None

    # data load
    test_lst, test_labels = load_dataset(isTest=True)

    ############################################################
    train_loss_epoch = []
    val_loss_epoch = []

    test_loader = load_dataloader('Test')


    print('starting train...', time.ctime())
    for epoch in range(Epoch):
        print('-'*110)
        print(f'Epoch {epoch+1}/{Epoch}, ', end='')

        train_loss, train_roc_auc = snp_train_dl(train_loader, train_lst, train_labels, model_mut, cri_ce, optimizer)
        val_loss, val_roc_auc = snp_test_dl('Val', val_loader, train_lst, train_labels, model_mut, cri_ce, thres_value)
        # if epoch % 3 == 0:
        #     test_loss, test_roc_auc = snp_test_dl('Test', test_loader, test_lst, test_labels, model_mut, cri_ce, thres_value)


        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)

        # early stop loss
        best_mut_model, best_val_loss, cur_wait = early_stopping_loss(model_mut, best_mut_model, val_loss, best_val_loss, cur_wait)

        if cur_wait == 100:
            break


    print('finished train. ', time.ctime())
    ##### 保存模型
    best_model_path = Model_PATH
    if IS_SAVE:
        print('save model path: ', Model_PATH)
        torch.save(best_mut_model.state_dict(), best_model_path)         #保存

    torch.cuda.empty_cache()
    # 测试
    test_loss, test_roc_auc = snp_test_dl('Test', test_loader, test_lst, test_labels, best_mut_model, cri_ce, thres_value)
    return best_mut_model


def snp_train_dl(train_loader, train_lst, train_labels, model_mut, cri_ce, optimizer):

    train_loss = []
    train_probs = []
    train_true_labels = []

    model_mut.train()

    for i, data in enumerate(train_loader):
        index = data
        feat_lst, true_label = get_feats(train_lst, train_labels, index)


        optimizer.zero_grad()

        output,tmp_out = model_mut(feat_lst)
        output = model_mut.sigmoid(output)

        # loss
        loss_ce = cri_ce(output, true_label)

        loss = loss_ce

        loss.backward()
        # nn.utils.clip_grad_norm_(model_mut.parameters(), 1)
        optimizer.step()

        # for name,param in model_mut.named_parameters():
        #     # print(name, param.grad)
        #     print(name)

        train_loss.append(loss.item())

        # 获取概率
        probs =  output.cpu().detach().numpy()
        train_probs.extend(probs)

        # 获取实际标签
        true_label_cpu = true_label.cpu().numpy()
        train_true_labels.extend(true_label_cpu)

    # 指标
    train_roc_auc = roc_auc_score(train_true_labels, train_probs, average='micro')
    train_loss = np.mean(train_loss)
    print(f'Train Loss: {train_loss:.4f}, AUC: {train_roc_auc:.4f}, ', end='')
    return train_loss, train_roc_auc



def snp_test_dl(data_type, test_loader, train_lst, train_labels, model_mut, cri_ce, thres_value):

    test_loss = []
    test_probs = []
    test_pred_labels = []
    test_true_labels = []

    with torch.no_grad():

        for i, data in enumerate(test_loader):
            index = data
            feat_lst, true_label = get_feats(train_lst, train_labels, index)

            output, tmp_out = model_mut(feat_lst)
            output = model_mut.sigmoid(output)

            # loss
            loss_ce = cri_ce(output, true_label)
            loss = loss_ce

            test_loss.append(loss.item())

            # 获取概率
            probs =  output.cpu().detach().numpy()
            test_probs.extend(probs)
            # 获取实际标签
            true_label_cpu = true_label.cpu().numpy()
            test_true_labels.extend(true_label_cpu)

            # 获取预测标签
            y_pred = np.zeros_like(true_label_cpu)  # 初始化为负类
            y_pred[probs[:] > thres_value] = 1  # 大于 0.7 的概率值为正类（1）
            y_pred[probs[:] < thres_value] = 0  # 小于 0.2 的概率值为负类（0）
            test_pred_labels.extend(y_pred)


    # 指标
    test_loss = np.mean(test_loss)

    test_roc_auc = roc_auc_score(test_true_labels, test_probs, average='micro')
    recall = recall_score(test_true_labels, test_pred_labels)
    precision = precision_score(test_true_labels, test_pred_labels, zero_division=1)

    if data_type == 'Test':
        print('\t\t', end='')


    print(f'{data_type} Loss: {test_loss:.4f}, AUC: {test_roc_auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}')

    if data_type == 'Test':
        return test_loss, test_roc_auc
    else:
        return test_loss, test_roc_auc




#################################################### ML
def test_func(train_lst, train_labels, best_mut_model):
    train_loader = load_dataloader('Train')
    val_loader = load_dataloader('Val')

    ############################################################################
    # ML
    ### ML train
    print('train by machine learning...')
    ml_models = snp_train_ml(train_lst, train_labels, best_mut_model, train_loader, val_loader)
    
    ### ML test
    print('test by machine learning...')

    for test_name in TEST_LST:
        print('=' * 100)
        snp_test_ml(ml_models, test_name, best_mut_model)


def extract_features(data_feats_list, data_labels, model_seq, train_loader):
    features_noadd = []
    features_af = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            index = data
            feats_lst, out_label = get_feats(data_feats_list, data_labels, index)

            _, train_tmp_out = model_seq(feats_lst)
            #### 添加特征
            _, _, scores, af, _, _, _, _, _ = feats_lst
            train_tmp_out = train_tmp_out.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            af = af.cpu().detach().numpy()

            features_noadd.extend(train_tmp_out)

            train_tmp_out_af = np.column_stack((train_tmp_out, scores, af))

            features_af.extend(train_tmp_out_af)


            labels.extend(out_label.cpu().detach().numpy())
    return features_noadd, features_af, labels



def snp_train_ml(data_feats_list, data_labels, model_seq, train_loader, val_loader):
    print('get feature by deep learning...')
    train_features, train_features_af, train_labels = extract_features(data_feats_list, data_labels, model_seq, train_loader)
    val_features, val_features_af, val_labels = extract_features(data_feats_list, data_labels, model_seq, val_loader)
    train_features.extend(val_features)
    train_features = np.array(train_features)

    train_features_af.extend(val_features_af)
    train_features_af = np.array(train_features_af)

    train_labels.extend(val_labels)
    train_labels = np.array(train_labels)
    train_labels = train_labels.reshape(-1)
    print('XGB with af train...')
    xgb_af = xgb.XGBClassifier(colsample_bytree=1, learning_rate=0.05, max_depth= 3, n_estimators=200, subsample=0.6, random_state=0)
    xgb_af.fit(train_features_af, train_labels)
    # return lgbm, lgbm_af, xgb_noaf, xgb_af
    return xgb_af




def snp_test_ml(ml_models, Test_Data, model_seq):
    xgb_af = ml_models
    print('Test for ', Test_Data)
    print('=' * 50)

    test_loader = load_dataloader('Test', Test_Data)
    test_data_feats_list, test_data_labels = load_dataset(True, Test_Data)

    test_features, test_features_af, test_labels = extract_features(test_data_feats_list, test_data_labels, model_seq, test_loader)
    test_features = np.array(test_features)
    test_features_af = np.array(test_features_af)
    test_labels = np.array(test_labels)
    test_labels = test_labels.reshape(-1)
    # 计算 AUC 分数
    if Test_Data == 'CSPD' or Test_Data == 'NSPD':
        if Test_Data == 'CSPD':
            no_train_index_path = '../data_loader/CSPD/binary/CSPD_test_index_noinMetaRNN.txt'

        elif Test_Data == 'NSPD':
            no_train_index_path = '../data_loader/NSPD/binary/NSPD_test_index_noinMetaRNN.txt'
        with open(no_train_index_path, 'r') as f:
            no_train_idx = [int(line.strip()) for line in f]
        no_train_idx = np.array(no_train_idx)
        

    xgb_af_scores = xgb_af.predict_proba(test_features_af)[:, 1]


    if Test_Data == 'CSPD' or Test_Data == 'NSPD':
        notrain_labels = test_labels[no_train_idx]
        notrain_xgb_af_scores = xgb_af_scores[no_train_idx]
    else:
        notrain_labels = test_labels
        notrain_xgb_af_scores = xgb_af_scores


    print('test data len:', notrain_labels.shape)

    ############################################# XGB af all
    print('\033[31m--- XGB with af all ---\033[0m')
    xgb_af_auc_score = roc_auc_score(notrain_labels, notrain_xgb_af_scores)
    print(f"XGBoost with af AUC: {xgb_af_auc_score:.4f}")

    ############################################################################## 保存数据
    if IS_SAVE_PROB:
        res = xgb_af_scores
        prob_save_path = '../../res_analyse/tmp_res/' + str(TRAIN_TIME) + '_' + str(Random_Seed) + '_' + Test_Data + '.npy'
        print(prob_save_path)
        np.save(prob_save_path, res)




def early_stopping_loss(model_mut, best_mut_model, val_loss, best_val_loss, cur_wait):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_mut_model = model_mut
        cur_wait = 0
    else:
        cur_wait += 1

        if cur_wait >= PATIENCE:    # early stopping
            print("Early stopping triggered.")
            cur_wait = 100

    return best_mut_model, best_val_loss, cur_wait
    


main()