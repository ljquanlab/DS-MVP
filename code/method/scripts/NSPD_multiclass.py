import argparse
import random
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score, roc_auc_score, average_precision_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
# Alpha = 1
# Beta = 1

# LR_STEP = 15
Threshold_value = 0.5


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
    label_path = '../../../feature/label/NSPD_multiclass.npy'
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
        train_index_path = '../data_loader/NSPD/multi_class/NSPD_multiclass_train_index.txt'
        with open(train_index_path, 'r') as f:
            train_index = [int(line.strip()) for line in f]
            train_index = torch.LongTensor(train_index)
        data_loader = DataLoader(dataset=train_index, batch_size=BATCH_SIZE, shuffle=True)


    elif data_type == 'Val':
        val_index_path = '../data_loader/NSPD/multi_class/NSPD_multiclass_train_index.txt'
        with open(val_index_path, 'r') as f:
            val_index = [int(line.strip()) for line in f]
            val_index = torch.LongTensor(val_index)
        data_loader = DataLoader(dataset=val_index, batch_size=BATCH_SIZE, shuffle=False)


    else:
        test_index_path = '../data_loader/NSPD/multi_class/NSPD_multiclass_train_index.txt'
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

    ############################################################################# load upstream model
    ################################## 修改部分层
    model_mut = DS_MVP_Model().to(device)
    # print(model_mut)
    # new_weights_dict = model_mut.state_dict()

    model_mut.load_state_dict(torch.load(Upstream_Model_PATH), False) #加载
    train_lst, train_labels = load_dataset()
    best_threshold = 0.5
    test_func(train_lst, train_labels, model_mut, best_threshold)



def test_func(train_lst, train_labels, best_mut_model, best_threshold):
    # train_loader = load_dataloader('Train')
    # val_loader = load_dataloader('Val')

    ############################################################################
    # ML
    ### ML train
    print('train by machine learning...')
    ### ML test
    print('test by machine learning...')
    val_thresholds_lst = []
    train_lst, train_labels = load_dataset()

    snp_test_ml(train_lst, train_labels, train_lst, train_labels, best_mut_model)




######################### ML
def extract_features(data_feats_list, data_labels, model_seq, train_loader):
    features = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
        # loop = tqdm(enumerate(train_loader), total=len(train_loader))
        # for i, data in loop:  
            index = data.to(device)
            feats_lst, out_label = get_feats(data_feats_list, data_labels, index)

            _, train_tmp_out = model_seq(feats_lst)
            #### 添加特征
            _, _, scores, af, _, _, _, _, _ = feats_lst
            train_tmp_out = train_tmp_out.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            af = af.cpu().detach().numpy()
            train_tmp_out = np.column_stack((train_tmp_out, scores, af))

            features.extend(train_tmp_out)

            # features.extend(train_tmp_out.cpu().detach().numpy())
            labels.extend(out_label.cpu().detach().numpy())


    return features, labels


def snp_test_ml(data_feats_list, data_labels, test_data_feats_list, test_data_labels, model_seq):
    train_loader = load_dataloader('Train')
    val_loader = load_dataloader('Val')
    test_loader = load_dataloader('Test')


    train_features, train_labels = extract_features(data_feats_list, data_labels, model_seq, train_loader)
    val_features, val_labels = extract_features(data_feats_list, data_labels, model_seq, val_loader)
    train_features.extend(val_features)
    train_labels.extend(val_labels)
    train_labels = np.array(train_labels)
    test_features, test_labels = extract_features(test_data_feats_list, test_data_labels, model_seq, test_loader)


    train_labels = np.array(train_labels)
    train_label_one = np.ones((len(train_labels), 1))
    train_label_one[train_labels[:, 0] == 1] = 0
    train_label_one[train_labels[:, 1] == 1] = 1
    train_label_one[train_labels[:, 2] == 1] = 2
    train_label_one[train_labels[:, 3] == 1] = 3

    test_labels = np.array(test_labels)
    test_label_one = np.ones((len(test_labels), 1))
    test_label_one[test_labels[:, 0] == 1] = 0
    test_label_one[test_labels[:, 1] == 1] = 1
    test_label_one[test_labels[:, 2] == 1] = 2
    test_label_one[test_labels[:, 3] == 1] = 3

    ###################################################################################
    # 
    #           XGB
    # 
    ###################################################################################


    dtrain = xgb.DMatrix(train_features, label=train_label_one)
    dtest = xgb.DMatrix(test_features, label=test_label_one)
    # model = xgb.XGBClassifier()
    # model.fit(train_features, train_label_one)
    # 设置XGBoost参数
    params = {
        'objective': 'multi:softprob',  # 使用softmax多分类
        'num_class': 4,                # 类别数
        'learning_rate':0.05,
        'max_depth': 3,                # 树的最大深度
        'eta': 0.1,                    # 学习率
        'subsample': 0.6,              # 训练数据的子采样比率
        'colsample_bytree': 1,       # 训练特征的子采样比率
        'eval_metric': 'mlogloss'      # 多分类log损失
    }

    # 训练XGBoost模型
    num_round = 100
    xgb_cf = xgb.train(params, dtrain, num_round)
        
    
    # 对测试集进行预测
    y_scores = xgb_cf.predict(dtest)
    
    # np.save('ndg_multiclass3_b_dl.npy', y_scores)

    y_pred = np.zeros(test_labels.shape)
    # 评估模型
    # 找到每一行的最大值的索引
    max_indices = np.argmax(y_scores, axis=1)
    # 将每一行的最大值置为 1
    y_pred[np.arange(len(y_scores)), max_indices] = 1


    y_pred = np.array(y_pred)
    y_pred_one = np.ones((len(y_pred), 1))
    y_pred_one[y_pred[:, 0] == 1] = 0
    y_pred_one[y_pred[:, 1] == 1] = 1
    y_pred_one[y_pred[:, 2] == 1] = 2
    y_pred_one[y_pred[:, 3] == 1] = 3


    ############## None
    precision = precision_score(test_labels, y_pred, average=None)
    recall = recall_score(test_labels, y_pred, average=None)
    f1 = f1_score(test_labels, y_pred, average=None)
    auc = roc_auc_score(test_labels, y_scores, average=None)
    ap = average_precision_score(test_labels, y_scores, average=None)

    ############## macro
    precision_macro = precision_score(test_labels, y_pred, average='macro')
    recall_macro = recall_score(test_labels, y_pred, average='macro')
    f1_macro = f1_score(test_labels, y_pred, average='macro')
    auc_macro = roc_auc_score(test_labels, y_scores, average='macro')
    ap_macro = average_precision_score(test_labels, y_scores, average='macro')
    ############## weighted
    precision_weighted = precision_score(test_labels, y_pred, average='weighted')
    recall_weighted = recall_score(test_labels, y_pred, average='weighted')
    f1_weighted = f1_score(test_labels, y_pred, average='weighted')
    auc_weighted = roc_auc_score(test_labels, y_scores, average='weighted')
    ap_weighted = average_precision_score(test_labels, y_scores, average='weighted')


    accuracy = accuracy_score(test_labels, y_pred)      # 总的
    conf_matrix = confusion_matrix(test_label_one, y_pred_one)
    class_report = classification_report(test_label_one, y_pred_one)
    
    print(f'ACC: {accuracy:.4f}')
    print("Confusion Matrix:\n", conf_matrix)
    # print("Classification Report:\n", class_report)
    print(f'               precision        recall          f1-score           auc            aupr')
    for i in range(test_labels.shape[1]):
        print(f' class{i}          {precision[i]:.4f}          {recall[i]:.4f}          {f1[i]:.4f}          {auc[i]:.4f}          {ap[i]:.4f} ')
    print(f' macro_avg       {precision_macro:.4f}          {recall_macro:.4f}          {f1_macro:.4f}          {auc_macro:.4f}          {ap_macro:.4f} ')
    print(f' weighted_avg    {precision_weighted:.4f}          {recall_weighted:.4f}          {f1_weighted:.4f}          {auc_weighted:.4f}          {ap_weighted:.4f} ')



    




main()