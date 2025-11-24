import argparse
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from utils_fp_downstream import *
from torch_geometric.loader import DataLoader
from model_gnn_fp_downstream import DPMoE
import torch.nn as nn
import math
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import random
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler


def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) # CPU
        torch.cuda.manual_seed(seed) # GPU
        torch.cuda.manual_seed_all(seed) # All GPU
        os.environ['PYTHONHASHSEED'] = str(seed) 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

def train(model, device, data_loader, optimizer, epoch, lam, DPMoE_variant):
    print('Training on {} samples...'.format(len(data_loader.dataset)))
    model.train()
    train_pred = torch.Tensor()
    train_y = torch.Tensor()
    total_loss = 0.0
    loss_values = []
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        if DPMoE_variant == 'weighted_sum':
            out, y, w, out_loss, y_loss, w_loss = model.forward_weighted_sum(data)
        elif DPMoE_variant == 'gumble_lb_loss':
            out, y, w, out_loss, y_loss, w_loss, lb_loss = model.forward_gumble_lb_loss(data)
        elif DPMoE_variant == 'gumble_adaptive_lb_loss':
            out, y, w, out_loss, y_loss, w_loss, lb_loss = model.forward_gumble_adaptive_lb_loss(data)

        pred_loss = nn.Sigmoid()(out_loss)
        pred = nn.Sigmoid()(out)
        train_pred = torch.cat((train_pred, torch.Tensor(pred.cpu().data.numpy())), 0)
        train_y = torch.cat((train_y, torch.Tensor(y.cpu().data.numpy())), 0)
        if DPMoE_variant == 'weighted_sum':
            loss = nn.BCELoss(weight=w_loss, reduction='mean')(pred_loss, y_loss)
        elif DPMoE_variant == 'gumble_lb_loss':
            loss = nn.BCELoss(weight=w_loss,reduction='mean')(pred_loss,y_loss) + lam*lb_loss
        elif DPMoE_variant == 'gumble_adaptive_lb_loss':
            loss = nn.BCELoss(weight=w_loss, reduction='mean')(pred_loss, y_loss) + lb_loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} \tLoss: {:.6f}'.format(epoch, total_loss/(batch_idx+1)))
    
    avg_loss = total_loss / len(data_loader)
    return train_pred, train_y,avg_loss

def predicting(model, device, data_loader, lam, DPMoE_variant):
    model.eval()     
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    val_loss = 0.0
    val_loss_values = []
    print('Make prediction for {} samples...'.format(len(data_loader.dataset)))
    with torch.no_grad():      
        for data in data_loader:
            data = data.to(device)
            if DPMoE_variant == 'weighted_sum':
                out, y, w, out_loss, y_loss, w_loss = model.forward_weighted_sum(data)
            elif DPMoE_variant == 'gumble_lb_loss':
                out, y, w, out_loss, y_loss, w_loss, lb_loss = model.forward_gumble_lb_loss(data)
            elif DPMoE_variant == 'gumble_adaptive_lb_loss':
                out, y, w, out_loss, y_loss, w_loss, lb_loss = model.forward_gumble_adaptive_lb_loss(data)
            pred = nn.Sigmoid()(out)
            pred_loss = nn.Sigmoid()(out_loss)
            pred = pred.to('cpu')
            y_ = y.to('cpu')
            total_preds = torch.cat((total_preds, pred), 0)
            total_labels = torch.cat((total_labels, y_), 0)
            if DPMoE_variant == 'weighted_sum':
                loss = nn.BCELoss(weight=w_loss,reduction='mean')(pred_loss,y_loss)
            elif DPMoE_variant == 'gumble_lb_loss':
                loss = nn.BCELoss(weight=w_loss, reduction='mean')(pred_loss, y_loss)+ lam * lb_loss
            elif DPMoE_variant == 'gumble_adaptive_lb_loss':
                loss = nn.BCELoss(weight=w_loss, reduction='mean')(pred_loss, y_loss) + lb_loss
            val_loss += loss.item()
    avg_loss = val_loss / len(data_loader)

    return total_preds.numpy().flatten(), total_labels.numpy().flatten(), avg_loss

def caculate_auc(array1,array2,m):
    M = len(array1)
    auc_list = []
    
    for i in range(m):
        sub_array1 = array1[i::m]
        sub_array2 = array2[i::m]
        non_999_indices = sub_array1 != 999
        sub_array1 = sub_array1[non_999_indices]
        sub_array2 = sub_array2[non_999_indices]
        
        if np.unique(sub_array1).size>1:
            auc = roc_auc_score(sub_array1,sub_array2)
            auc_list.append(auc)
           
    return np.mean(auc_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual pathway MOE')
    parser.add_argument('--path', default='down_task')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--random_seed', default=9, type=int)
    parser.add_argument('--task', default='bace', type=str,
                        choices=['bace', 'bbbp', 'clintox', 'sider', 'tox21'])
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--lam', type=float, default=0.01) # for gumble
    parser.add_argument('--p1_num', default=2, type=int) # for gumble
    parser.add_argument('--p2_num', default=2, type=int) # for gumble
    parser.add_argument('--DPMoE_variant', default='weighted_sum', type=str,
                        choices=['weighted_sum', 'gumble_lb_loss', 'gumble_adaptive_lb_loss'])

    args = parser.parse_args()     
    print(args)
    
    batch_size, epochs = args.batch_size, args.epochs   
    task, random_seed = args.task, args.random_seed
    
    
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed) # CPU
        torch.cuda.manual_seed(seed) # GPU
        torch.cuda.manual_seed_all(seed) # All GPU
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
    
    set_seed(random_seed)

    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    lam = args.lam
    p1_num, p2_num = args.p1_num, args.p2_num
    DPMoE_variant = args.DPMoE_variant


    LOG_INTERVAL = 20
    
    clr_tasks = {'bbbp': 1, 'hiv': 1, 'bace': 1, 'tox21': 12, 'clintox': 2, 'sider': 27, 'MUV': 17, 'toxcast':617, 'PCBA':128}
    task_num = clr_tasks[task]

    train_data = TestbedDataset(root=args.path, dataset='train', task=task)
    valid_data = TestbedDataset(root=args.path, dataset='valid', task=task)
    test_data = TestbedDataset(root=args.path, dataset='test', task=task)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    x_dim = train_data[0].x.shape[1]
    pubchem_dim = train_data[0].pubchem.shape[0]
    maccs_dim = train_data[0].maccs.shape[0]
    erg_dim = train_data[0].erg.shape[0]
    ecfp_dim = train_data[0].ecfp.shape[0]
    fp_dim = pubchem_dim+maccs_dim+erg_dim+ecfp_dim
    edge_dim = train_data[0].edge_attr.shape[1]
    model = DPMoE(x_input=x_dim, fp_input=fp_dim, edge_dim=edge_dim, output=task_num, p1_num=p1_num, p2_num=p2_num, DPMoE_variant=DPMoE_variant).to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': 1e-4, 'weight_decay': 1e-2}])

    scheduler = lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)

    save_file ='{}'.format(task)
    if not os.path.exists('results/'+ task):
        os.makedirs('results/'+ task)
    save_name = 'results/' + task
    result_file_name = save_name+'/'+save_file+'_result.csv'
    valid_AUCs = save_name+'/'+save_file+'_validAUCs.txt'
    test_AUCs = save_name+'/'+save_file+'_testAUCs.txt'
    train_AUCs = save_name+'/'+save_file+'_trainAUCs.txt'
    model_file_name =save_name+'/'+save_file+'_encoder.pkl'
    AUCs = ('Epoch\tAUC')

    with open(valid_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    with open(test_AUCs, 'w') as f:
        f.write(AUCs + '\n')
    with open(train_AUCs, 'w') as f:
        f.write(AUCs + '\n')
 

    best_auc = 0
    stopping_monitor = 0
    train_losses = []
    valid_losses = []
    test_losses = []
    for epoch in range(epochs+1):
        train_pred,train_y,test_loss= train(model, device, train_data_loader, optimizer, epoch + 1, lam, DPMoE_variant)
        train_losses.append(test_loss)
        valid_pred,valid_true,val_loss= predicting(model,device,valid_data_loader, lam, DPMoE_variant)
        valid_losses.append(val_loss)
        test_pred,test_true,test_loss = predicting(model,device,test_data_loader, lam, DPMoE_variant)
        test_losses.append(test_loss)
        
        if (epoch + 0) % 5 == 0:
            train_auc = caculate_auc(train_y,train_pred,task_num)
            AUCs = [epoch,train_auc]
            save_AUCs(AUCs,train_AUCs)
            
            print('train_AUC:',train_auc)

            valid_auc = caculate_auc(valid_true,valid_pred,task_num)
            AUCs = [epoch, valid_auc]
            print('valid_AUC: ', AUCs)

            if best_auc < valid_auc:
                best_auc = valid_auc
                stopping_monitor = 0
                print('best_auc：', best_auc)
                save_AUCs(AUCs, valid_AUCs)
                print('save model weights')
                torch.save(model.state_dict(), model_file_name)
            else:
                stopping_monitor += 1
            if stopping_monitor > 0:
                print('stopping_monitor:', stopping_monitor)
            if stopping_monitor > 20:
                break
    
    model.load_state_dict(torch.load(model_file_name))
    test_pred, test_true, loss = predicting(model, device,test_data_loader,lam, DPMoE_variant)
    print('pred value：', test_pred)
    print('true value：', test_true)
    print('loss value：', loss)
    test_pred_value = save_name+'/'+save_file+'_pred.txt'
    test_true_value = save_name+'/'+save_file+'_true.txt'
    save_AUCs(test_pred, test_pred_value)
    save_AUCs(test_true, test_true_value)
    
    test_auc = caculate_auc(test_true, test_pred, task_num)
    AUCs = [0, test_auc]
    print(task,random_seed,'test_AUC: ', AUCs)

    save_AUCs(AUCs, test_AUCs)
    
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.plot(test_losses, label='Test loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_name+'/'+task+'_loss.png')
    plt.show()
    