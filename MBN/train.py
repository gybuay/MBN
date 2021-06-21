import torch, torch.nn as nn
from torch.autograd import Variable
import numpy as np
import traceback
import argparse
from tqdm import tqdm
import random

from config import Config
from MBN import MBN_Model
from metrics import str_metric, get_NDCG, get_hit, make_metrics
from DataHelper import batch_generator
from utils import adjust_learning_rate, read_data, logging, cache_logging, save_model, remove_model
from tensorboardX import SummaryWriter
writer = SummaryWriter('piclog12')
# tensorboard --logdir=./piclog12


def train_network(network, datas, num_epoch=10, target_task=2, display_step=1):
    print("epochs: ", num_epoch, "target_task", target_task)
    history = []
    save_epoch = []
    try:
        for epoch in range(num_epoch):
            g = batch_generator(datas, target_task, hp.batch_size, hp.have_tasks, train=True, order=True, time_format={"tianchi": "%Y-%m-%d %H", "ijcai": "%m%d"}[hp.dataset_name])
            cnt = 0
            v_loss = 0.
            t_loss = 0.

            pbar = tqdm(g)
            pbar.set_description("%s" % '[Train] Epoch = %d [loss = x.xx]' % (epoch + 1))

            for user_batch, item_batch, mask_batch, copy_batch in pbar:

                user_batch_ix = Variable(torch.LongTensor(user_batch)).to(device)
                item_batch_ix = Variable(torch.LongTensor(item_batch)).to(device)
                mask_batch_ix = Variable(torch.FloatTensor(mask_batch)).to(device)
                copy_batch_ix = Variable(torch.LongTensor(copy_batch)).to(device)

                logp_seqs = network(user_batch_ix, item_batch_ix, copy_batch_ix)
                logp_seq = logp_seqs[0]

                # cross entropy loss
                item_batch_ix = item_batch_ix[hp.have_tasks.index(target_task), :, :, :]
                actual_next_tokens = item_batch_ix[:, 1:, :]
                actual_next_num = torch.sum((item_batch_ix > 0).type_as(mask_batch_ix), 2)

                actual_next_num = actual_next_num + (actual_next_num == 0).type_as(actual_next_num)

                predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, 1:, None] / actual_next_num[:, 1:, None]

                logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, :])
                logp_next = logp_next * (actual_next_tokens > 0).type_as(logp_next)
                loss = -logp_next.sum() / mask_batch_ix[:, 1:].sum()
                v_loss = loss.cpu().data.numpy()
                cnt += 1
                
                if np.isnan(v_loss):
                    raise RuntimeError()

                # train with backprop
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 5)
                opt.step()

                log_info = '[Train] Epoch = %d [loss = %.4f]' % \
                   (epoch + 1, v_loss.item())
                pbar.set_description("%s" % log_info)

                writer.add_scalar('Train/TrainLossBatch_%d' % epoch, v_loss.item(), cnt)
                t_loss += v_loss.item()

            writer.add_scalar('Train/TrainLossEpoch', v_loss.item(), epoch)
            logging("[Train] epochs=%d, train_loss=%.6f" % (epoch + 1, t_loss/cnt))

            v_train_loss = v_loss
            logging("[EVAL] epochs=%d, cnt=%d" % (epoch + 1, cnt))
            if epoch < hp.display_start and epoch != 0:
                save_epoch.append(epoch)
                if len(save_epoch) > hp.max_to_keep:
                    rm_epoch = save_epoch[0]
                    remove_model(hp.log_dir, rm_epoch)
                    save_epoch = save_epoch[1:]
                save_model(network, hp, hp.log_dir, epoch, opt)
                continue

            v_test_loss, metricses = evaluate_hit(network, datas, train=False, target_task=target_task)

            writer.add_scalar('Eval/Loss', v_test_loss, epoch)
            writer.add_scalar('Eval/Recall', metricses[-1]['recall-micro'], epoch)
            writer.add_scalar('Eval/NDCG', metricses[-1]['ndcg-macro'], epoch)

            history.append([epoch + 1, v_train_loss, v_test_loss, metricses])
            if max(history,
                   key=lambda x: x[3][0]["recall-macro"])[3][0]["recall-macro"] == metricses[0]["recall-macro"]:
                hp.history = history
                save_epoch.append(epoch)
                if len(save_epoch) > hp.max_to_keep:
                    rm_epoch = save_epoch[0]
                    remove_model(hp.log_dir, rm_epoch)
                    save_epoch = save_epoch[1:]
                save_model(network, hp, hp.log_dir, epoch, opt)

    except KeyboardInterrupt:
        traceback.print_exc()
        hp.history = history
        save_model(network, hp, hp.log_dir + "/final", num_epoch, opt)
        return history
    evaluate_hit(network, datas, train=False, target_task=target_task)
    hp.history = history
    save_model(network, hp, hp.log_dir + "/final", num_epoch, opt)
    return history


def evaluate_hit(network, datas, train=True, target_task=2, loss_id=0):
    topk_list = hp.topk_list
    hits_all = [[] for _ in range(len(topk_list))]
    ndcgs_all = [[] for _ in range(len(topk_list))]
    actuals_all = []
    metricses = []

    network.eval()
    nums = []
    losses = []
    with torch.no_grad():
        g = batch_generator(datas, target_task, hp.batch_size, hp.have_tasks, train=train, order=True, time_format={"tianchi": "%Y-%m-%d %H", "ijcai": "%m%d"}[hp.dataset_name])
        cnt = 0

        pbar = tqdm(g)
        pbar.set_description("%s" % '[Eval] [loss = x.xx]')

        for user_batch, item_batch, mask_batch, copy_batch in pbar:
            user_batch_ix = Variable(torch.LongTensor(user_batch)).to(device)
            item_batch_ix = Variable(torch.LongTensor(item_batch)).to(device)
            mask_batch_ix = Variable(torch.FloatTensor(mask_batch)).to(device)
            copy_batch_ix = Variable(torch.LongTensor(copy_batch)).to(device)

            logp_seqs = network(user_batch_ix, item_batch_ix, copy_batch_ix)
            logp_seq = logp_seqs[loss_id]
            # compute loss
            item_batch_ix = item_batch_ix[hp.have_tasks.index(target_task), :, :, :]
            actual_next_num = torch.sum((item_batch_ix > 0).type_as(mask_batch_ix), 2)
            actual_next_num = actual_next_num + (actual_next_num == 0).type_as(actual_next_num)
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, 1:, None] / actual_next_num[:, 1:, None]
            actual_next_tokens = item_batch_ix[:, 1:, :]
            logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, :])
            logp_next = logp_next * (actual_next_tokens > 0).type_as(logp_next)
            # torch.nn.MultiLabelMarginLoss
            loss = -logp_next.sum() / mask_batch_ix[:, 1:].sum()
            nums.append(mask_batch_ix[:, 1:].sum().cpu().data.numpy())
            v_loss = loss.cpu().data.numpy()
            cnt += 1
            losses.append(float(v_loss))

            log_info = '[Eval] [loss = %.4f]' % \
                (v_loss.item())
            pbar.set_description("%s" % log_info)

            # get topk
            logp_seq_arr = logp_seq.cpu().data.numpy()
            batch_idx, batch_seq_idx = np.where(mask_batch)
            for bi, bsi in zip(batch_idx, batch_seq_idx):
                y_pred = np.argsort(-logp_seq_arr[bi, bsi - 1, :])[:max(topk_list)]
                y_true = [item for item in item_batch[hp.have_tasks.index(target_task), bi, bsi, :] if item != 0]
                for topk_idx, topk in enumerate(topk_list):
                    hits_all[topk_idx].append(get_hit(y_true, y_pred, topk))
                    ndcgs_all[topk_idx].append(get_NDCG(y_true, y_pred, topk))
                actuals_all.append(len(y_true))

    r_loss = float(np.sum(losses) / np.sum(nums))
    logging("[EVAL] target_task=%d, eval_loss=%.6f, cnt=%d" % (target_task, r_loss, cnt))
    for topk_idx, topk in enumerate(topk_list):
        metrics = make_metrics(hits_all[topk_idx], actuals_all, ndcgs_all[topk_idx], topk)
        print("[EVAL] %s@%d" % (["test", "train"][int(train)], topk), str_metric(metrics))
        metricses.append(metrics)
    return r_loss, metricses

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sum the integers at the command line')
    parser.add_argument('-d', '--dataset',  type=str, help='tianchi or ijcai15')
    parser.add_argument('-b', '--basket_pool',  type=str, help='ave, max, attn')
    parser.add_argument('-r', '--rnn_mode',  type=str, help='rnn, gru, lstm')
    parser.add_argument('-copy', '--copy_mode',  type=int, help='0: normal, 1: only repeat, 2 only generate')
    parser.add_argument('-cross', '--cross_mode',  type=int, help='rnn cross or not')
    parser.add_argument('-tasks', '--have_tasks',  type=str, help='buy,favor,click,cart, split by ,')
    parser.add_argument('-c', '--cuda', type=int, help='gpu number')
    parser.add_argument('-mix', '--mix_mode', type=int, help='mix mode 0: sum, 1: max', default=1)
    parser.add_argument('-lr', '--learning_rate', type=float, help='mix mode 0: sum, 1: max', default=0.001)
    parser.add_argument('-dim', '--dimension', type=int, help='dimension', default=100)
    parser.add_argument('--random_seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    parser.add_argument('-mtr', '--meta_rnn', type=int, help='1:meta rnn or 0:not', default=1)
    parser.add_argument('-wd', '--weight_decay', type=float, help='l2 norm weight', default=0)
    parser.add_argument('-ep', '--epochs', type=int, help='epochs', default=200)
    parser.add_argument('-drop', '--dropout', type=float, help='dropout', default=0.1)
    parser.add_argument('-use_drop', '--use_drop', type=int, help='1:meta rnn or 0:not', default=0)
    parser.add_argument('-behavior_type', '--behavior_type', type=int, help='1:attn or 0:concat', default=0)
    parser.add_argument('-res', '--residual', type=int, help='1:yes or 0:no', default=0)
    parser.add_argument('-cp', '--checkpoint', type=int, help='checkpoint', default=0)
    args = parser.parse_args()

    hp = Config(args.dataset)
    hp.pool_mode = args.basket_pool
    hp.rnn_mode = args.rnn_mode
    hp.temp_mode = False
    hp.repeat = args.copy_mode
    hp.mix_mode = args.mix_mode
    hp.lr = args.learning_rate
    hp.cross_mode = True if args.cross_mode == 1 else False
    hp.have_tasks = [int(_c) for _c in args.have_tasks.split(",")]
    hp.n_tasks = len(hp.have_tasks)

    hp.d_item = args.dimension
    hp.d_rnn = args.dimension
    hp.d_attn = args.dimension

    hp.meta_rnn = args.meta_rnn

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    hp.epochs = args.epochs
    hp.dropout = args.dropout
    hp.use_drop = args.use_drop
    hp.behavior_type = args.behavior_type
    hp.residual = args.residual
    hp.cuda = args.cuda
    env_cuda = "cuda:%d" % hp.cuda
    
    TIME_FORMAT = {"tianchi": "%Y-%m-%d %H", "ijcai": "%m%d"}[hp.dataset_name]
    print(hp)

    device = torch.device(env_cuda if torch.cuda.is_available() else "cpu")
    torch.nn.Module.dump_patches = True
    seqs = read_data(hp.file_dir + "/seqs_%d_enum.pkl", 4)
    network = MBN_Model(hp).to(device)
    print(network.layer_wise_parameters())
    print(network)

    old_epochs = 0
    opt = torch.optim.Adam(network.parameters(), lr=hp.lr, weight_decay=args.weight_decay)
    train_network(network, seqs, hp.epochs - old_epochs, target_task=hp.target_task, display_step=hp.display_step)