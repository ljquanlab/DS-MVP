import argparse
import random
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score,precision_score
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, hamming_loss
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')



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
Model_PATH = '../opt_model/checkpoint_card_' + str(TRAIN_TIME) +'.pth'

Epoch = 300
BATCH_SIZE = 16
LR = 1e-3
W_D = 0
PATIENCE = 30
NumClass=7




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
    label_path = '../../../feature/label/CSPD_multilabel.npy'

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
        train_index_path = '../data_loader/CSPD/multi_label/CSPD_multilabel_train_index.txt'

        with open(train_index_path, 'r') as f:
            train_index = [int(line.strip()) for line in f]
            train_index = torch.LongTensor(train_index)
        data_loader = DataLoader(dataset=train_index, batch_size=BATCH_SIZE, shuffle=True)


    elif data_type == 'Val':
        val_index_path = '../data_loader/CSPD/multi_label/CSPD_multilabel_val_index.txt'
        with open(val_index_path, 'r') as f:
            val_index = [int(line.strip()) for line in f]
            val_index = torch.LongTensor(val_index)
        data_loader = DataLoader(dataset=val_index, batch_size=BATCH_SIZE, shuffle=False)


    else:
        test_index_path = '../data_loader/CSPD/multi_label/CSPD_multilabel_test_index.txt'
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
    print('upstream model path: ', Upstream_Model_PATH)

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
    print(train_layer)
    for name, param in model_up.named_parameters():
        if any(name.startswith(prefix) for prefix in train_layer):
            # print(name)
            params.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    optimizer = optim.Adam(params, lr=LR, weight_decay=W_D)

    num_classes = NumClass

    print(model_mut)
    print(model_mut.classfier[0].weight.data)
    #############################################################################

    print('train Model PATH:', Model_PATH)

    weights = torch.tensor([1/324,1/461,1/15,1/21,1/16,1/90, 1/1373]).to(device)  # 类别1的权重为1，类别2的权重为10
    cri_ce = nn.BCEWithLogitsLoss(pos_weight=weights).to(device)


    # earlystop
    cur_wait = 0
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

        if epoch % 3 == 0:
            test_loss, test_roc_auc = snp_test('Test', train_lst, train_labels, model_mut, cri_ce, thres_value)


    print('finished train. ', time.ctime())
    ##### 保存模型
    best_model_path = Model_PATH
    # torch.save(best_mut_model.state_dict(), best_model_path)         #保存

    torch.cuda.empty_cache()
    # best_mut_model
    print(best_mut_model.classfier[0].weight.data)
    # exit(0)
    # 测试
    test_loss, test_roc_auc = snp_test('Test', train_lst, train_labels, best_mut_model, cri_ce, thres_value)
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
    # val_loader = load_dataloader('Val')
    train_loss = []
    train_probs = []
    train_true_labels = []

    model_mut.train()

    for i, data in enumerate(train_loader):
        index = data.to(device)
        feat_lst, true_label = get_feats(train_lst, train_labels, index)

        optimizer.zero_grad()

        output,tmp_out = model_mut(feat_lst)
        out_lst = torch.Tensor(np.zeros(true_label.shape)).to(device)
        loss = 0
        for i in range(NumClass):
            output_single = model_mut.classfier(tmp_out)
            output_single = model_mut.sigmoid(output_single)
            out_lst[:, i] = (output_single.squeeze(-1))
            # loss += loss_ce_single
        
        loss = cri_ce(out_lst, true_label)
        
        # loss = loss/NumClass
        loss.backward()
        # nn.utils.clip_grad_norm_(model_mut.parameters(), 1)
        optimizer.step()
        # for name,param in model_mut.named_parameters():
        #     # print(name, param.grad)
        #     print(name)

        train_loss.append(loss.item())

        # 获取概率

    train_loss = np.mean(train_loss)
    
    print(f'Train Loss: {train_loss:.4f}, ', end='')
    train_roc_auc = 0
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

            probs = np.zeros(true_label.shape)

            output,tmp_out = model_mut(feat_lst)
            out_lst = torch.Tensor(np.zeros(true_label.shape)).to(device)
            loss = 0
            for i in range(NumClass):
                output_single = model_mut.classfier(tmp_out)
                output_single = model_mut.sigmoid(output_single)
                probs[:, i] = output_single.squeeze(-1).cpu().detach().numpy()
                out_lst[:, i] = (output_single.squeeze(-1))
                # loss += loss_ce_single
            
            loss = cri_ce(out_lst, true_label)
            ##################################
            test_loss.append(loss.item())



            # 获取概率
            test_probs.extend(probs)
            # 获取实际标签
            true_label_cpu = true_label.cpu().numpy()
            test_true_labels.extend(true_label_cpu)


    # 指标

    test_loss = np.mean(test_loss)

    test_true_labels = np.array(test_true_labels)
    test_probs = np.array(test_probs)
    # test_pred_labels = np.array(test_pred_labels)
    test_pred_labels = (test_probs > thres_value).astype(int)  # 将概率转换为类别预测


    test_roc_auc = roc_auc_score(test_true_labels, test_probs, average='micro')
    recall = recall_score(test_true_labels, test_pred_labels,  average='micro')
    precision = precision_score(test_true_labels, test_pred_labels, zero_division=1,  average='micro')

    if data_type == 'Test':
        print('\t\t', end='')



    print(f'{data_type} Loss: {test_loss:.4f}, AUC: {test_roc_auc:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}')


    if data_type == 'Test':
        return test_loss, test_roc_auc
    else:
        return test_loss, test_roc_auc


######################### ML
def extract_features(data_feats_list, data_labels, model_seq, train_loader):
    features = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
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
    test_labels = np.array(test_labels)


    n_labels = NumClass
    print(train_labels.shape[1])
    # 训练XGBoost模型
    models = []
    for i in range(train_labels.shape[1]):
        model = xgb.XGBClassifier()
        model.fit(train_features, train_labels[:, i])
        models.append(model)


    ###############################################
    # 进行预测
    y_scores = np.zeros(test_labels.shape)
    for i, model in enumerate(models):
        y_scores[:, i] = model.predict_proba(test_features)[:, 1]

    # np.save('./card_multi6_b_prob_dl.npy', y_scores)
    y_pred = np.zeros(test_labels.shape)

    y_pred = (y_scores > 0.5).astype(int)  # 将概率转换为类别预测



    # 计算每个标签的指标
    label_metrics = {}
    for i in range(n_labels):
        accuracy = accuracy_score(test_labels[:, i], y_pred[:, i])
        f1 = f1_score(test_labels[:, i], y_pred[:, i])
        mcc = matthews_corrcoef(test_labels[:, i], y_pred[:, i])

        roc_auc = roc_auc_score(test_labels[:, i], y_scores[:, i])
        ap = average_precision_score(test_labels[:, i], y_scores[:, i])
        label_metrics[f'label_{i}'] = {'accuracy': accuracy, 'f1_score': f1, 'roc_auc': roc_auc, 'mcc':mcc, 'aupr':ap}

    # 输出每个标签的指标
    for label, metrics in label_metrics.items():
        print(f"Metrics for {label}:   Accuracy: {metrics['accuracy']:.4f}   F1 Score: {metrics['f1_score']:.4f}   MCC: {metrics['mcc']:.4f}   ROC AUC: {metrics['roc_auc']:.4f}   AUPR: {metrics['roc_auc']:.4f}")
        # print(f"{label}:   ROC AUC: {metrics['roc_auc']:.4f}")





    # 计算多标签分类的指标
    accuracy = accuracy_score(test_labels, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    p = precision_score(y_true=test_labels, y_pred=y_pred, average='samples')
    print(f'precision Score (samples): {p:.4f}')

    recall = recall_score(y_true=test_labels, y_pred=y_pred, average='samples')
    print(f'recall Score (samples): {recall:.4f}')

    f1_macro = f1_score(test_labels, y_pred, average='samples')
    print(f'F1 Score (samples): {f1_macro:.4f}')

    hamming = hamming_loss(test_labels, y_pred)
    print(f'Hamming Loss: {hamming:.4f}')




main()