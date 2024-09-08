import argparse
import datetime
import math
import pickle
import random
import time

from torch.utils.data import DataLoader

from MCGNN import MCGNN, train_test, trans_to_cuda
from graph.collate import gnn_collate_fn, collate
from graph.graph_construction import seq_to_hetero_graph
from log import logger
from utils.util import *


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


argparser = argparse.ArgumentParser()
argparser.add_argument('--model_name', default='MCGNN', type=str, help='model name')
argparser.add_argument('--emb_size', default=100, type=int, help='embedding size')
argparser.add_argument('--gpu', default=0, type=int, help='gpu id')
# argparser.add_argument('--max_length', default=10, type=int, help='max session length')
argparser.add_argument('--dataset',default='Tmall',help='diginetica/Nowplaying/Tmall')
argparser.add_argument('--batch', default=2, type=int, help='batch size')
argparser.add_argument('--epoch', default=20, type=int, help='total epochs')
argparser.add_argument('--patience',default=3,type=int,help='early stopping patience')
argparser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
argparser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
argparser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
argparser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
argparser.add_argument('--save_flag', default=False, type=bool,help='save checkpoint or not')
argparser.add_argument('--fdrop', default=0.2, type=float, help='feature drop')
argparser.add_argument('--adrop', default=0.0, type=float, help='attention drop')
argparser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')
argparser.add_argument('--num_layers', default=1, type=int, help='gnn layers')
argparser.add_argument('--graph_feature_select', default='gated',help='last/gated/mean')
argparser.add_argument('--validation', action='store_true', help='validation')
argparser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
argparser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
argparser.add_argument('--neighbor_num', type=int, default=5, help='.')
argparser.add_argument('--dropout1', type=float, default=0.0, help='.')
argparser.add_argument('--dropout2', type=float, default=0.0, help='.')
argparser.add_argument('--seed', type=int, default=2023, help='.')
argparser.add_argument('--norm', type=bool, default=False, help='.')
args = argparser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')



def main():
    init_seed(args.seed)
    print(args)
    logger.error(args)


    if args.dataset == 'diginetica':
        num_item = 43098
        num_cat = 996


    elif args.dataset == 'Nowplaying':
        num_item = 60417
        num_cat = 11462


    elif args.dataset == 'Tmall':
        num_item = 40728
        num_cat = 712
#         args.dropout_local = 0.5

    elif args.dataset == 'yoochoose1_64_new':
        num_item = 17746
        num_cat = 36
#         args.dropout_local = 0.5
    else:
        num_item = 9
        num_cat = 3

    train_data = pickle.load(open('datasets/' + args.dataset + '/train.txt', 'rb'))
    if args.validation:
        train_data, valid_data = split_validation(train_data, args.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + args.dataset + '/test.txt', 'rb'))

    category = pickle.load(open('datasets/' + args.dataset + '/category.txt', 'rb'))
    category = handle_category(category)

    data_statistics(train_data, test_data, category)

    train_data = Data(train_data, category)
    test_data = Data(test_data, category)


    train_loader = DataLoader(train_data, batch_size = args.batch,shuffle=True,num_workers=0, collate_fn=collate, pin_memory = True)
    test_loader = DataLoader(test_data,batch_size=args.batch, shuffle=False,num_workers=0, collate_fn=collate,pin_memory = True)



    model = trans_to_cuda(MCGNN(args,
                  num_item,
                  num_cat,
                  device,
                  batch_norm=True,
                  feat_drop=args.fdrop,
                  attention_drop=args.adrop).to(device))

    start = time.time()
    best_result_k10 = [0, 0]
    best_result_k20 = [0, 0]
    best_result_k30 = [0, 0]
    best_result_k40 = [0, 0]
    best_result_k50 = [0, 0]

    best_epoch_k10 = [0, 0]
    best_epoch_k20 = [0, 0]
    bad_counter_k20 = bad_counter_k10 = 0


    for epoch in range(args.epoch):
        print('-------------------------------------------------------')
        logger.error('-------------------------------------------------------')
        print('epoch: ', epoch, " lr: " , model.optimizer.param_groups[0]['lr'])
        logger.info("epoch: %s, lr: %s." % (epoch, model.optimizer.param_groups[0]['lr']))

        hit_k10, mrr_k10, hit_k20, mrr_k20, hit_k30, mrr_k30, hit_k40, mrr_k40, hit_k50, mrr_k50 = train_test(model,
                                                                                                              train_loader,
                                                                                                             test_loader)

        flag_k10 = 0
        if hit_k10 >= best_result_k10[0]:
            best_result_k10[0] = hit_k10
            best_epoch_k10[0] = epoch
            flag_k10 = 1
        if mrr_k10 >= best_result_k10[1]:
            best_result_k10[1] = mrr_k10
            best_epoch_k10[1] = epoch
            flag_k10 = 1

        print("\n")
        print('Current Result:')
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f' % (hit_k10, mrr_k10))
        print('Best Result:')
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result_k10[0], best_result_k10[1], best_epoch_k10[0], best_epoch_k10[1]))
        bad_counter_k10 += 1 - flag_k10

        logger.error("\n")
        logger.error('Current Result:')
        logger.error('\tRecall@10:\t%.4f\tMMR@10:\t%.4f' % (hit_k10, mrr_k10))
        logger.error('Best Result:')
        logger.error('\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result_k10[0], best_result_k10[1], best_epoch_k10[0], best_epoch_k10[1]))

        flag_k20 = 0
        if hit_k20 >= best_result_k20[0]:
            best_result_k20[0] = hit_k20
            best_epoch_k20[0] = epoch
            flag_k20 = 1
        if mrr_k20 >= best_result_k20[1]:
            best_result_k20[1] = mrr_k20
            best_epoch_k20[1] = epoch
            flag_k20 = 1
        print("\n")
        print('Current Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit_k20, mrr_k20))
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result_k20[0], best_result_k20[1], best_epoch_k20[0], best_epoch_k20[1]))

        logger.error("\n")
        logger.error('Current Result:')
        logger.error('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit_k20, mrr_k20))
        logger.error('Best Result:')
        logger.error('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result_k20[0], best_result_k20[1], best_epoch_k20[0], best_epoch_k20[1]))

        bad_counter_k20 += 1 - flag_k20

        if hit_k30 >= best_result_k30[0]:
            best_result_k30[0] = hit_k30
        if mrr_k30 >= best_result_k30[1]:
            best_result_k30[1] = mrr_k30
        print('Best @30 Result:')
        print('\tRecall@30:\t%.4f\tMMR@30:\t%.4f' % (
            best_result_k30[0], best_result_k30[1]))

        logger.error('Best @30 Result:')
        logger.error('\tRecall@30:\t%.4f\tMMR@30:\t%.4f' % (
            best_result_k30[0], best_result_k30[1]))

        if hit_k40 >= best_result_k40[0]:
            best_result_k40[0] = hit_k40
        if mrr_k40 >= best_result_k40[1]:
            best_result_k40[1] = mrr_k40
        print('Best @40 Result:')
        print('\tRecall@40:\t%.4f\tMMR@40:\t%.4f' % (
            best_result_k40[0], best_result_k40[1]))

        logger.error('Best @40 Result:')
        logger.error('\tRecall@40:\t%.4f\tMMR@40:\t%.4f' % (
            best_result_k40[0], best_result_k40[1]))

        if hit_k50 >= best_result_k50[0]:
            best_result_k50[0] = hit_k50
        if mrr_k50 >= best_result_k50[1]:
            best_result_k50[1] = mrr_k50
        print('Best @50 Result:')
        print('\tRecall@50:\t%.4f\tMMR@50:\t%.4f' % (
            best_result_k50[0], best_result_k50[1]))

        logger.error('Best @50 Result:')
        logger.error('\tRecall@50:\t%.4f\tMMR@50:\t%.4f' % (
            best_result_k50[0], best_result_k50[1]))

        if ((bad_counter_k20 >= args.patience) and (bad_counter_k10 >= args.patience)):
            break

    print('-------------------------------------------------------')
    logger.error('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))
    logger.error("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
