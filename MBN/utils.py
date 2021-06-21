import time
import os
import pickle
import torch

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR 0.9 every 1 epoch"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * (0.99 ** (epoch // 1))

def read_data(file_path, n_tasks):
    ret = []
    for i in range(n_tasks):
        logging("[READ] read => %s" % (file_path % i))
        with open(file_path % i, "rb") as rf:
            seqs = pickle.load(rf)
        new_seqs = dict()
        for user, user_seq in seqs.items():
            new_user_seq = []
            for basket_i, time_str, is_pred, is_test in user_seq:
                new_user_seq.append([[item + 1 for item in basket_i],
                                     time_str, is_pred, is_test])
            new_seqs[user] = new_user_seq
        ret.append(new_seqs)
    return ret

def logging(log_str):
    print("[INFO %s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), log_str))


def cache_logging(log_str):
    print("[INFO %s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), log_str), end="\r")
    pass

def save_model(network, history, file_dir, number, opt):
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    logging("[SAVE] save => %s/model_%d.pkl" % (file_dir, number))
    torch.save(network, "%s/model_%d.pkl" % (file_dir, number))
    torch.save({
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            }, "%s/model_%d.pkl" % (file_dir, number))
    logging("[SAVE] save=> %s/history_%d.pkl" % (file_dir, number))
    with open("%s/history_%d.pkl" % (file_dir, number), "wb") as fw:
        pickle.dump(history, fw)


def remove_model(file_dir, number):
    model_fp = "%s/model_%d.pkl" % (file_dir, number)
    history_fp = "%s/history_%d.pkl" % (file_dir, number)
    logging("[REMOVE] remove => %s" % model_fp)
    if os.path.exists(model_fp):
        os.remove(model_fp)
    logging("[REMOVE] remove => %s" % history_fp)
    if os.path.exists(history_fp):
        os.remove(history_fp)