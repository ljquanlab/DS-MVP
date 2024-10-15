import argparse
import random
import time
from datetime import datetime
import torch
import torch.nn as nn
# import torch.optim as optim
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import recall_score,precision_score, roc_auc_score, average_precision_score, precision_recall_curve
import xgboost as xgb

from models import *     # 7.31 2
Upstream_Model_PATH = '../opt_model/20240731_084324.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print(device)


# 获取当前时间
now = datetime.now()
formatted_time = now.strftime('%Y%m%d_%H%M%S')
print(formatted_time)
TRAIN_TIME = formatted_time


# TRAIN_TIME = time.ctime()
Model_PATH = '../opt_model/ndg_' + str(TRAIN_TIME) +'.pth'

Epoch = 300
BATCH_SIZE = 32
LR = 1e-3
W_D = 0
PATIENCE = 30


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(1024)



def load_dataset():
    # load data

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
    data_list = []
    for i in range(len(feats_path_list)):
        data = np.load(feats_path_list[i])
        data = torch.Tensor(np.array(data)).to(device)
        data_list.append(data)


    label = np.load(label_path)
    label =  torch.Tensor(np.array(label)).to(device)
    return data_list, label




def load_dataloader(data_type='Test'):
    #(2)加载索引
    if data_type == 'Train':
        train_index_path = '../data_loader/CSPD/binary/CSPD_finetune_train_index.txt'
        with open(train_index_path, 'r') as f:
            train_index = [int(line.strip()) for line in f]
            train_index = torch.LongTensor(train_index)
        data_loader = DataLoader(dataset=train_index, batch_size=BATCH_SIZE, shuffle=True)


    elif data_type == 'Val':
        val_index_path = '../data_loader/CSPD/binary/CSPD_finetune_val_index.txt'
        with open(val_index_path, 'r') as f:
            val_index = [int(line.strip()) for line in f]
            val_index = torch.LongTensor(val_index)
        data_loader = DataLoader(dataset=val_index, batch_size=BATCH_SIZE, shuffle=False)


    else:
        test_index_path = '../data_loader/CSPD/binary/CSPD_finetune_test_index.txt'
        with open(test_index_path, 'r') as f:
            test_index = [int(line.strip()) for line in f]
            test_index = torch.LongTensor(test_index)
        data_loader = DataLoader(dataset=test_index, batch_size=BATCH_SIZE, shuffle=False)

    return data_loader


def get_feats(data_feats_list, data_labels, index):

    data_lst = []
    for feat in data_feats_list:
        data_lst.append(feat[index])


    # label
    out_label = data_labels[index]
    return data_lst, out_label


def main():
    ############################################################################# load upstream model
    parser = argparse.ArgumentParser(description='Fine-tune based on model file.')

    # 添加位置参数
    parser.add_argument('--opt_model_path', type=str, default=Upstream_Model_PATH, help='Model file path')

    args = parser.parse_args()


    try:    
        print('upstream model path: ', args.opt_model_path)
        model_setups(args.opt_model_path)
    except Exception as e:
        print(f"An error occurred: {e}")


def model_setups(Upstream_Model_PATH):
    ################################## 修改部分层
    model_mut = DS_MVP_Model().to(device)
    # print(model_mut)
    new_weights_dict = model_mut.state_dict()

    model_up = DS_MVP_Model()
    model_up.load_state_dict(torch.load(Upstream_Model_PATH), False) #加载
    weights_dict = model_up.state_dict()

    # print(new_weights_dict)
    # print(weights_dict)

    # 加载预训练的参数
    for k in weights_dict.keys():
        if k in new_weights_dict.keys() and not k.startswith('sigmoid'):  
            new_weights_dict[k] = weights_dict[k]
    
    model_mut.load_state_dict(new_weights_dict)
    #####################################################################################
    # way 1选择要训练的参数
    params = []
    # train_layer = ['classfier']
    select_layer = ['seq_express', 'scores_af_express', 'mgw_shape_express', 'gene_express', 'aa_express', 'classfier']
    # select_idx = [5]
    # select_idx = [0,5]
    # select_idx = [1,5]
    # select_idx = [2,5]
    # select_idx = [3,5]
    select_idx = [4,5]
    train_layer = [select_layer[i] for i in select_idx]


    # train_layer = ['classfier', 'scores_af_express', 'gene_express', 'mgw_shape_express']       ######## ok
    for name, param in model_mut.named_parameters():
        if any(name.startswith(prefix) for prefix in train_layer):
            # print(name)
            params.append(param)
        else:
            param.requires_grad = False
    # print(model_mut.classfier[0].weight.data)
    
    # exit(0)
    optimizer = optim.Adam(params, lr=LR, weight_decay=W_D)

    print(model_mut)
    #############################################################################

    print('train Model PATH:', Model_PATH)

    cri_ce = nn.BCELoss()
    print(train_layer)

    # earlystop
    cur_wait = 0
    best_roc_auc = 0
    best_val_loss = np.inf
    best_mut_model = None
    thres_value = 0.5


    # data load
    train_lst, train_labels = load_dataset()


    train_loss_epoch = []
    val_loss_epoch = []


    print('starting train...', time.ctime())
    for epoch in range(Epoch):
        print('-'*110)
        print(f'Epoch {epoch+1}/{Epoch}, ', end='')

        train_loss, train_roc_auc = snp_train(train_lst, train_labels, model_mut, cri_ce, optimizer)
        val_loss, val_roc_auc = snp_test('Val', train_lst, train_labels, model_mut, cri_ce, thres_value)

        # early stop loss
        best_mut_model, best_val_loss, cur_wait = early_stopping_loss(model_mut, best_mut_model, val_loss, best_val_loss, cur_wait)

        train_loss_epoch.append(train_loss)
        val_loss_epoch.append(val_loss)

        if cur_wait == 100:
            break

        # if epoch % 3 == 0:
        #     test_loss, test_roc_auc = snp_test('Test', train_lst, train_labels, model_mut, cri_ce, thres_value)


    print('finished train. ', time.ctime())
    ##### 保存模型
    best_model_path = Model_PATH
    # torch.save(best_mut_model.state_dict(), best_model_path)         #保存

    torch.cuda.empty_cache()
    # print(best_mut_model.classfier[0].weight.data)
    # 测试
    test_loss, test_roc_auc = snp_test('Test', train_lst, train_labels, best_mut_model, cri_ce, thres_value)
    # show_tsne(train_lst, train_labels, test_lst, test_labels, model_mut)
    # ML
    snp_test_ml(train_lst, train_labels, train_lst, train_labels, best_mut_model)





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
    


def snp_train(train_lst, train_labels, model_mut, cri_ce, optimizer):
    train_loader = load_dataloader('Train')

    train_loss = []
    train_probs = []
    train_true_labels = []

    model_mut.train()

    for i, data in enumerate(train_loader):
        index = data.to(device)
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



def snp_test(data_type, train_lst, train_labels, model_mut, cri_ce, thres_value):
    test_loader = load_dataloader(data_type)

    test_loss = []
    test_probs = []
    test_pred_labels = []
    test_true_labels = []

    with torch.no_grad():

        for i, data in enumerate(test_loader):
            index = data.to(device)
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
    recall = recall_score(test_true_labels, test_pred_labels,  average='micro')
    precision = precision_score(test_true_labels, test_pred_labels, zero_division=1,  average='micro')

    if data_type == 'Test':
        print('\t\t', end='')

    print(f'{data_type} Loss: {test_loss:.4f}, AUC: {test_roc_auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}')

    return test_loss, test_roc_auc



######################### ML

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
            _, _, scores, af, _, _, gene_embed, _, _ = feats_lst
            train_tmp_out = train_tmp_out.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            af = af.cpu().detach().numpy()
            gene_embed = gene_embed.cpu().detach().numpy()

            features_noadd.extend(train_tmp_out)

            train_tmp_out_af = np.column_stack((train_tmp_out, scores, af))

            features_af.extend(train_tmp_out_af)


            labels.extend(out_label.cpu().detach().numpy())
    return features_noadd, features_af, labels




def snp_test_ml(data_feats_list, data_labels, test_data_feats_list, test_data_labels, model_seq):
    train_loader = load_dataloader('Train')
    val_loader = load_dataloader('Val')


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


    test_loader = load_dataloader('Test')
    # test_data_feats_list, test_data_labels = load_dataset()

    test_features, test_features_af, test_labels = extract_features(test_data_feats_list, test_data_labels, model_seq, test_loader)
    test_features = np.array(test_features)
    test_features_af = np.array(test_features_af)
    test_labels = np.array(test_labels)
    test_labels = test_labels.reshape(-1)

    xgb_af_scores = xgb_af.predict_proba(test_features_af)[:, 1]

    xgb_af_auc_score = roc_auc_score(test_labels, xgb_af_scores)
    print(xgb_af_auc_score)
    xgb_af_aupr_score = average_precision_score(test_labels, xgb_af_scores)
    print(xgb_af_aupr_score)



main()
