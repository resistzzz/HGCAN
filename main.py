import os
import argparse
import pickle
import time
import datetime
from model import *
import torch
import numpy as np
from data import *
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import sys

def init_seed(seed):
    #  sets a random seed to generate a specified random number in numpy
    np.random.seed(seed)
    # sets a random seed to generate a specified random number in pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='YC', help='sample/YC/DIG/JD')

'''Training basic parameters'''
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_epoch', type=list, default=[2, 4, 6, 8], help='the epoch which the learning rate decay')
parser.add_argument('--patience', type=int, default=5)

'''Model hyperparameters'''
parser.add_argument('--topk', type=list, default=[20], help='topk recommendation')  # Top-20 Recommendation
parser.add_argument('--toph', type=list, default=[10], help='topk recommendation')  # Top-10 Recommendation
parser.add_argument('--save_path', default='model_save', help='save model root path')
parser.add_argument('--save_epochs', default=[2, 4, 6, 8], type=list)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--routing_iter', type=int, default=4)
parser.add_argument('--hop', type=int, default=2)  # 1 or 2 or 3
parser.add_argument('--sample_num', type=int, default=12)
parser.add_argument('--dropout_gnn', type=float, default=0.2)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--beta', type=float, default=0.001)
parser.add_argument('--aba', type=int, default=0)   #When there is no ablation experiment, it is set to 0
parser.add_argument('--cate_select', type=int, default=5)  #
parser.add_argument('--dice_threshold', type=float, default=1.0)#Proportion of users who gave feedback
opt = parser.parse_args()

class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


file_name = './log/'+str(opt.dataset)+'/cate_select='+str(opt.cate_select)+\
            ', hop='+str(opt.hop)+',threshold='+str(opt.dice_threshold)+',beat='+str(opt.beta)+'_'+str(datetime.datetime.now())[-5:]+'.txt'
sys.stdout = Logger(file_name, sys.stdout)
sys.stderr = Logger(file_name, sys.stderr)  # redirect std err, if necessary

print(opt)

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')
# device = torch.device( 'cpu')

if opt.save_path is not None and opt.dataset != 'sample':
    save_path = opt.save_path + '/' + opt.dataset
    save_dir = save_path + '/' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print('save dir: ', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def main():
    t0 = time.time()
    init_seed(1022)
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    train_data_cat = pickle.load(open('datasets/' + opt.dataset + '/train_cat.txt', 'rb'))
    test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    test_data_cat = pickle.load(open('datasets/' + opt.dataset + '/test_cat.txt', 'rb'))
    cate2item = pickle.load(open('datasets/' + opt.dataset + '/cate2item.txt', 'rb'))
    item2cate = pickle.load(open('datasets/' + opt.dataset + '/item2cate.txt', 'rb'))
    param = pickle.load(open('datasets/' + opt.dataset + '/parm' + '.pkl', 'rb'))
    adj_items = pickle.load(open('datasets/' + opt.dataset + '/adj_' + str(opt.n_sample) + '.pkl', 'rb'))
    weight_items = pickle.load(open('datasets/' + opt.dataset + '/num_' + str(opt.n_sample) + '.pkl', 'rb'))
    adj_cate = pickle.load(open('datasets/' + opt.dataset + '/adj_cate_' + str(opt.n_sample) + '.pkl', 'rb'))
    weight_cate = pickle.load(open('datasets/' + opt.dataset + '/num_cate_' + str(opt.n_sample) + '.pkl', 'rb'))
    adj_c2i = pickle.load(open('datasets/' + opt.dataset + '/adj_c2i_' + str(opt.n_sample) + '.pkl', 'rb'))
    weight_c2i = pickle.load(open('datasets/' + opt.dataset + '/num_c2i_' + str(opt.n_sample) + '.pkl', 'rb'))

    num_items, max_length, num_cate = param['num_items'], param['max_length'], param['num_cate']


    train_data = Data(train_data, num_items)
    test_data = Data(test_data, num_items)
    train_data_cat = Data(train_data_cat, num_items)
    test_data_cat = Data(test_data_cat, num_items)


    train_slices = train_data.generate_batch(opt.batch_size)
    test_slices = test_data.generate_batch(opt.batch_size)
    train_slices_cat = train_data_cat.generate_batch(opt.batch_size)
    test_slices_cat = test_data_cat.generate_batch(opt.batch_size)

    cate_item_matirx = np.zeros((num_cate, num_items))
    for i in cate2item:
        for j in cate2item[i]:
            cate_item_matirx[i][j] = 1

    adj_items, weight_items = handle_adj(adj_items, weight_items, num_items, opt.sample_num)
    adj_cate, weight_cate = handle_adj(adj_cate, weight_cate, num_cate, opt.sample_num)
    adj_c2i, weight_c2i = handle_adj(adj_c2i, weight_c2i, num_cate, opt.sample_num)
    adj_i2c = np.array(list(item2cate.values())).reshape(-1,1)

    model = HGCAN(opt, num_items, num_cate, max_length, adj_items, weight_items, adj_cate, weight_cate, adj_c2i, weight_c2i, adj_i2c, device)
    model = model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_dc_epoch, gamma=opt.lr_dc)

    best_result = {}
    best_epoch = {}
    best_result10 = {}
    best_epoch10 = {}
    for k in opt.topk:
        best_result[k] = [0, 0]
        best_epoch[k] = [0, 0]
    for k in opt.toph:
        best_result10[k] = [0, 0]
        best_epoch10[k] = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epochs):
        st = time.time()
        print('-------------------------------------------')
        print('epoch: ', epoch)

        hit, mrr, hit10, mrr10 = train_test(model, train_data, train_data_cat, test_data, test_data_cat,
                                            train_slices, train_slices_cat, test_slices, test_slices_cat,
                                            optimizer, max_length, cate2item)
        if opt.save_path is not None and epoch in opt.save_epochs and opt.dataset != 'sample':
            save_file = save_dir + '/epoch-' + str(epoch) + '.pt'
            torch.save(model, save_file)
            print('save success! :)')

        flag = 0
        for k in opt.topk:
            if hit[k] > best_result[k][0]:
                best_result[k][0] = hit[k]
                best_epoch[k][0] = epoch
                flag = 1
            if mrr[k] > best_result[k][1]:
                best_result[k][1] = mrr[k]
                best_epoch[k][1] = epoch
                flag = 1
            print('Hit@%d:\t%0.4f %%\tMRR@%d:\t%0.4f %%\t[%0.2f s]' % (k, hit[k], k, mrr[k], (time.time() - st)))
        for k in opt.toph:
            if hit10[k] > best_result10[k][0]:
                best_result10[k][0] = hit10[k]
                best_epoch10[k][0] = epoch
                flag = 1
            if mrr10[k] > best_result10[k][1]:
                best_result10[k][1] = mrr10[k]
                best_epoch10[k][1] = epoch
                flag = 1
            print('Hit@%d:\t%0.4f %%\tMRR@%d:\t%0.4f %%\t[%0.2f s]' % (k, hit10[k], k, mrr10[k], (time.time() - st)))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
        scheduler.step()

    print('------------------best result-------------------')
    for k in opt.topk:
        print('Best Result: Hit@%d: %0.4f %%\tMRR@%d: %0.4f %%\t[%0.2f s]' %
              (k, best_result[k][0], k, best_result[k][1], (time.time() - t0)))
        print('Best Epoch: Hit@%d: %d\tMRR@%d: %d\t[%0.2f s]' % (
            k, best_epoch[k][0], k, best_epoch[k][1], (time.time() - t0)))
    for k in opt.toph:
        print('Best Result: Hit@%d: %0.4f %%\tMRR@%d: %0.4f %%\t[%0.2f s]' %
              (k, best_result10[k][0], k, best_result10[k][1], (time.time() - t0)))
        print('Best Epoch: Hit@%d: %d\tMRR@%d: %d\t[%0.2f s]' % (
            k, best_epoch10[k][0], k, best_epoch10[k][1], (time.time() - t0)))
    print('------------------------------------------------')
    print('Run time: %0.2f s' % (time.time() - t0))


def train_test(model, train_data, train_data_cat, test_data, test_data_cat,
               train_slices, train_slices_cat, test_slices, test_slices_cat, optimizer, max_length, cate2item):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = []
    total_rec_loss = []
    total_cate_loss = []
    t0 = time.time()
    tr_hit, tr_mrr = {}, {}

    for k in opt.topk:
        tr_hit[k] = []
        tr_mrr[k] = []

    count_cate = []
    for index in train_slices:
        optimizer.zero_grad()

        scores_cate, scores_items, targets_cat, targets_items = forward(model, index, train_data, train_data_cat,
                                                                        max_length)

        cp = scores_cate.topk(opt.cate_select)[1] + 1
        mask = torch.zeros_like(scores_items)
        mask = mask.to(device)
        #===================Cate Conversation===================================================
        for i in range(scores_cate.size(0)):
            if targets_cat[i] in cp[i]:
                count_cate = np.append(count_cate, 1)
            elif targets_cat[i]  not in cp[i]:
                count_cate = np.append(count_cate, 0)

            dice = np.random.rand(1)
            if dice <= opt.dice_threshold:
                if targets_cat[i] in cp[i]:
                    item_index = cate2item[targets_cat[i].cpu().item()]
                    item_index = torch.tensor(item_index).to(device)
                    mask[i] = 1.0
                    mask[i, item_index - 1] = 0.0
                else:
                    mask[i] = 0.0
                    for c in cp[i]:
                        item_index = cate2item[c.cpu().item()]
                        item_index = torch.tensor(item_index).to(device)
                        mask[i, item_index - 1] = 1.0
        mask = mask * -9e15
        scores_items = scores_items + mask

        cate_loss = model.cate_loss_function(scores_cate, targets_cat - 1)
        item_loss = model.item_loss_function(scores_items, targets_items - 1)
        loss = opt.beta * cate_loss + item_loss
        loss.backward()

        optimizer.step()

        total_cate_loss.append(cate_loss.item())
        total_rec_loss.append(item_loss.item())
        total_loss.append(loss.item())

        for k in opt.topk:
            predict = scores_items.cpu().topk(k)[1]
            predict = predict.cpu()
            for pred, target in zip(predict, targets_items.cpu()):
                tr_hit[k].append(np.isin(target - 1, pred))
                if len(np.where(pred == target - 1)[0]) == 0:
                    tr_mrr[k].append(0)
                else:
                    tr_mrr[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))

    for k in opt.topk:
        tr_hit[k] = np.mean(tr_hit[k]) * 100
        tr_mrr[k] = np.mean(tr_mrr[k]) * 100

        print('Hit_tr@%d:\t%0.4f %%\tMRR_tr@%d:\t%0.4f %%\t' % (k, tr_hit[k], k, tr_mrr[k]))

    print('Cate:\t%.3f\tlr:\t%0.6f' % (np.mean(total_cate_loss), optimizer.state_dict()['param_groups'][0]['lr']))
    print('Item:\t%.3f\tlr:\t%0.6f' % (np.mean(total_rec_loss), optimizer.state_dict()['param_groups'][0]['lr']))
    print('All:\t%.3f\tlr:\t%0.6f' % (np.mean(total_loss), optimizer.state_dict()['param_groups'][0]['lr']))
    print('correct ', np.mean(count_cate))
    print('----------------')
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = {}, {}
    hit10, mrr10 = {}, {}
    for k in opt.topk:
        hit[k] = []
        mrr[k] = []
    for k in opt.toph:
        hit10[k] = []
        mrr10[k] = []

    te_count_cate = []
    te_total_loss = []
    te_total_rec_loss = []
    te_total_cate_loss = []

    with torch.no_grad():
        for index in test_slices:

            te_scores_cate, te_scores_items, te_targets_cat, tes_targets_items = forward(model,
                                                                                         index, test_data,
                                                                                         test_data_cat, max_length)
            # ===================Cate Conversation===================================================
            cp = te_scores_cate.topk(opt.cate_select)[1] + 1
            mask = torch.zeros_like(te_scores_items)
            mask = mask.to(device)
            for i in range(te_scores_cate.size(0)):

                if te_targets_cat[i] in cp[i]:
                    te_count_cate = np.append(te_count_cate, 1)
                elif te_targets_cat[i] not in cp[i]:
                    te_count_cate = np.append(te_count_cate, 0)

                dice = np.random.rand(1)
                if dice <= opt.dice_threshold:
                    if te_targets_cat[i] in cp[i]:
                        item_index = cate2item[te_targets_cat[i].cpu().item()]
                        item_index = torch.tensor(item_index).to(device)
                        mask[i] = 1.0
                        mask[i, item_index - 1] = 0.0
                    else:
                        mask[i] = 0.0
                        for c in cp[i]:
                            item_index = cate2item[c.cpu().item()]
                            item_index = torch.tensor(item_index).to(device)
                            mask[i, item_index - 1] = 1.0
            mask = mask * -9e15
            te_scores_items = te_scores_items + mask

            for k in opt.topk:
                predict = te_scores_items.cpu().topk(k)[1]
                predict = predict.cpu()
                for pred, target in zip(predict, tes_targets_items.cpu()):
                    hit[k].append(np.isin(target - 1, pred))
                    if len(np.where(pred == target - 1)[0]) == 0:
                        mrr[k].append(0)
                    else:
                        mrr[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))
            for k in opt.toph:
                predict = te_scores_items.cpu().topk(k)[1]
                predict = predict.cpu()
                for pred, target in zip(predict, tes_targets_items.cpu()):
                    hit10[k].append(np.isin(target - 1, pred))
                    if len(np.where(pred == target - 1)[0]) == 0:
                        mrr10[k].append(0)
                    else:
                        mrr10[k].append(1 / (np.where(pred == target - 1)[0][0] + 1))

            te_cate_loss = model.cate_loss_function(te_scores_cate, te_targets_cat - 1)
            te_item_loss = model.item_loss_function(te_scores_items, tes_targets_items - 1)
            te_loss = opt.beta * te_cate_loss + te_item_loss

            te_total_cate_loss.append(te_cate_loss.item())
            te_total_rec_loss.append(te_item_loss.item())
            te_total_loss.append(te_loss.item())

        for k in opt.topk:
            hit[k] = np.mean(hit[k]) * 100
            mrr[k] = np.mean(mrr[k]) * 100
            print('Hit1@%d:\t%0.4f %%\tMRR1@%d:\t%0.4f %%\t' % (k, hit[k], k, mrr[k]))
        for k in opt.toph:
            hit10[k] = np.mean(hit10[k]) * 100
            mrr10[k] = np.mean(mrr10[k]) * 100
            print('Hit1@%d:\t%0.4f %%\tMRR1@%d:\t%0.4f %%\t' % (k, hit10[k], k, mrr10[k]))

        print('pre', np.mean(te_count_cate))
        print('Cate:\t%.3f\t' % (np.mean(te_total_cate_loss)))
        print('Item:\t%.3f\t' % (np.mean(te_total_rec_loss)))
        print('All:\t%.3f\t' % (np.mean(te_total_loss)))

        return hit, mrr, hit10, mrr10


def forward(model, index, data, data_cat, max_length):
    inp_sess, inp_sess_padding, inp_sess_mask, targets_items, lengths = data.get_slice_sess_mask(index)
    inp_sess_cat, inp_sess_cat_padding, inp_sess_cat_mask, targets_cat, cate_lengths = data_cat.get_slice_sess_mask(
        index)
    item_cate_matrix, cate_matrix, item_cate_mask_inf, cate_mask_inf, query_matrix, reverse_positon_idx1 = data_cat.get_item_padding_by_cate(
        opt.batch_size, inp_sess, inp_sess_cat)
    reverse_positon_idx = data.get_reverse_position(inp_sess_padding, lengths)

    item_cate_matrix = torch.LongTensor(item_cate_matrix).to(device)
    cate_matrix = torch.LongTensor(cate_matrix).to(device)
    query_matrix = torch.LongTensor(query_matrix).to(device)
    item_cate_mask_inf = torch.FloatTensor(item_cate_mask_inf).to(device)
    cate_mask_inf = torch.FloatTensor(cate_mask_inf).to(device)
    targets_cat = torch.LongTensor(targets_cat).to(device)
    targets_items = torch.LongTensor(targets_items).to(device)
    reverse_positon_idx = torch.LongTensor(reverse_positon_idx).to(device)
    reverse_positon_idx1 = torch.LongTensor(reverse_positon_idx1).to(device)
    inp_sess_padding = torch.LongTensor(inp_sess_padding).to(device)
    # inp_sess_mask = torch.FloatTensor(inp_sess_mask).to(device)
    inp_sess_cat_padding = torch.LongTensor(inp_sess_cat_padding).to(device)
    lengths = torch.LongTensor(lengths).to(device)

    scores_cate, scores_items = model(inp_sess_padding , inp_sess_cat_padding, lengths, item_cate_matrix,
                                      cate_matrix, item_cate_mask_inf, cate_mask_inf, query_matrix, reverse_positon_idx,
                                      reverse_positon_idx1)

    return scores_cate, scores_items, targets_cat, targets_items


if __name__ == '__main__':
    main()


