from datetime import datetime as dt
import time
import json
import pandas as pd
import pickle
import numpy as np
import os
# import collections
import copy


def logging(log_str):
    print("[INFO %s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), log_str))


def concat_logs(input_dir, input_fns, output_dir, first_line=True):
    output_fp = "%s/logs.csv" % output_dir
    fw = open(output_fp, "w")

    for input_fn in input_fns:
        input_fp = "%s/%s" % (input_dir, input_fn)
        logging("[READ] read => %s" % input_fp)
        fs = True
        for line in open(input_fp, "r"):
            if first_line and fs:
                fs = False
                continue
            fw.write(line)
    fw.close()
    logging("[SAVE] save => %s" % output_fp)


def logs2seqs(input_fp, output_dir, first_line=False):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    data_info = {_i: dict() for _i in range(4)}
    seqs_types = [{} for _ in range(4)]

    # read
    logging("[READ] read => %s" % input_fp)
    for line in open(input_fp, "r"):
        if first_line:
            first_line = False
            continue
        line_list = line.strip().split(LINE_SEG)
        user_str = line_list[USER_IDX].strip()
        item_str = line_list[ITEM_IDX].strip()
        time_str = line_list[TIME_IDX]
        action = int(line_list[TYPE_IDX].strip())
        seqs_types[action].setdefault(user_str, dict()) # add user_str:{}
        seqs_types[action][user_str].setdefault(time_str, set()) # add time_str:()
        seqs_types[action][user_str][time_str].add(item_str) # add items that have the same time to a basket

    # stat
    min_time_dt = None
    max_time_dt = None
    for action, seqs in enumerate(seqs_types):
        user_log_count = {}
        user_date_count = {}
        item_log_count = {}
        for user_str, user_seq in seqs.items():
            new_user_seq = []
            user_log_count[user_str] = 0
            for time_str, basket in user_seq.items():
                new_user_seq.append([list(basket), time_str])
                user_log_count[user_str] += len(basket)
                for item_str in basket:
                    item_log_count.setdefault(item_str, 0)
                    item_log_count[item_str] += 1
            new_user_seq = sorted(new_user_seq, key=lambda x: dt.strptime(x[1], TIME_FORMAT))
            seqs[user_str] = new_user_seq
            user_date_count[user_str] = len(new_user_seq)
            if min_time_dt is None:
                min_time_dt = dt.strptime(new_user_seq[0][1], TIME_FORMAT)
                max_time_dt = dt.strptime(new_user_seq[-1][1], TIME_FORMAT)
            else:
                min_time_dt = min(min_time_dt, dt.strptime(new_user_seq[0][1], TIME_FORMAT))
                max_time_dt = max(max_time_dt, dt.strptime(new_user_seq[-1][1], TIME_FORMAT))
        data_info[action]["n_logs"] = sum(user_log_count.values())
        data_info[action]["n_users"] = len(user_log_count)
        data_info[action]["n_items"] = len(item_log_count)
        data_info[action]["n_baskets"] = sum(user_date_count.values())
        data_info[action]["from"] = dt.strftime(min_time_dt, TIME_FORMAT)
        data_info[action]["to"] = dt.strftime(max_time_dt, TIME_FORMAT)
        save_pkl("%s/seqs_%d.pkl" % (output_dir, action), seqs)

    save_json("%s/data_info.json" % output_dir, data_info, display=True)
    logging("[END] logs2seqs finished! ")


def save_pd(_data, name1, name2, _fp):
    logging("[SAVE] save => %s" % _fp)
    pd.DataFrame({name1: list(_data.keys()),
                  name2: list(_data.values())}).to_csv(_fp, index=False)


def save_json(_fp, datas, display=False):
    logging("[SAVE] save => %s" % _fp)
    with open(_fp, "wb") as fw:
        text = json.dumps(datas, indent=4, ensure_ascii=False)
        fw.write(text.encode("utf-8"))
        if display:
            print(text)


def save_pkl(_fp, datas):
    logging("[SAVE] save => %s" % _fp)
    with open(_fp, "wb") as fw:
        pickle.dump(datas, fw)


def count_seqs(seqs):
    n_cur_logs = 0
    user_log_count, item_log_count, user_date_count = dict(), dict(), dict()
    for user_str, user_seq in seqs.items():
        user_log_count[user_str] = 0
        user_date_count[user_str] = len(user_seq)
        for basket, time_str in user_seq:
            n_cur_logs += len(basket)
            user_log_count[user_str] += len(basket)
            for item_str in basket:
                item_log_count.setdefault(item_str, 0)
                item_log_count[item_str] += 1
    return n_cur_logs, user_log_count, item_log_count, user_date_count


def filter_seqs(seqs, user_log_flag, item_log_flag):
    new_seqs = dict()
    for user_str, user_seq in seqs.items():
        if user_str not in user_log_flag:
            continue
        new_user_seq = []
        for basket, time_str in user_seq:
            new_basket = list(filter(lambda x: x in item_log_flag, basket))
            if len(new_basket) > 0:
                new_user_seq.append([new_basket, time_str])
        if len(new_user_seq) > 0:
            new_seqs[user_str] = new_user_seq
    return new_seqs


def filter_seqs_dream(seqs_fp, output_dir, user_limit=10, item_limit=10, action=1): # 2,2,3
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logging("filter follow dream: %d %d" % (user_limit, item_limit))
    logging("[READ] read => %s" % (seqs_fp % action))
    with open(seqs_fp % action, "rb") as rf:
        seqs = pickle.load(rf)

    n_cur_logs, user_log_count, item_log_count, user_date_count = count_seqs(seqs)
    n_before_logs = n_cur_logs
    logging("[FILTER] n_cur_logs=%d" % n_cur_logs)
    n_filter = 0
    while True:
        logging("[FILTER] filter=%d, n_before_logs=%d" % (n_filter, n_cur_logs))
        n_filter += 1
        user_date_flag = dict(filter(lambda x: x[1] >= 2, user_date_count.items()))
        user_log_flag = dict(filter(lambda x: x[1] >= user_limit, user_log_count.items()))
        item_log_flag = dict(filter(lambda x: x[1] >= item_limit, item_log_count.items()))
        user_log_flag = set(user_date_flag) & set(user_log_flag)
        seqs = filter_seqs(seqs, user_log_flag, item_log_flag)
        # save_pkl("%s/seqs_%d.pkl" % (output_dir, n_filter), seqs)
        n_cur_logs, user_log_count, item_log_count, user_date_count = count_seqs(seqs)
        if n_cur_logs == n_before_logs:
            break
        else:
            n_before_logs = n_cur_logs
    save_pkl("%s/seqs_%d_filter_dream.pkl" % (output_dir, action), seqs)
    for new_action in [0, 1, 2]:  # todo notice!
        new_seqs_fp = seqs_fp % new_action
        logging("[READ] read => %s" % new_seqs_fp)
        with open(new_seqs_fp, "rb") as rf:
            seqs = pickle.load(rf)
        seqs = filter_seqs(seqs, user_log_flag, item_log_flag)
        save_pkl("%s/seqs_%d_filter_dream.pkl" % (output_dir, new_action), seqs)
    logging("[END] filter_seqs_dream finished! ")


def filter_seqs_binn(seqs_fp, output_dir, user_limit=3, item_limit=3, user_date=10, action=1):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logging("filter follow dream: %d %d" % (user_limit, item_limit))
    logging("[READ] read => %s" % (seqs_fp % action))
    with open(seqs_fp % action, "rb") as rf:
        seqs = pickle.load(rf)

    n_cur_logs, user_log_count, item_log_count, user_date_count = count_seqs(seqs)
    n_before_logs = n_cur_logs
    logging("[FILTER] n_cur_logs=%d" % n_cur_logs)
    n_filter = 0
    while True:
        logging("[FILTER] filter=%d, n_before_logs=%d" % (n_filter, n_cur_logs))
        n_filter += 1
        user_date_flag = dict(filter(lambda x: x[1] >= user_date, user_date_count.items()))
        user_log_flag = dict(filter(lambda x: x[1] >= user_limit, user_log_count.items()))
        item_log_flag = dict(filter(lambda x: x[1] >= item_limit, item_log_count.items()))
        user_log_flag = set(user_date_flag) & set(user_log_flag)
        seqs = filter_seqs(seqs, user_log_flag, item_log_flag)
        # save_pkl("%s/seqs_%d.pkl" % (output_dir, n_filter), seqs)
        n_cur_logs, user_log_count, item_log_count, user_date_count = count_seqs(seqs)
        if n_cur_logs == n_before_logs:
            break
        else:
            n_before_logs = n_cur_logs
    save_pkl("%s/seqs_%d_filter_dream.pkl" % (output_dir, action), seqs)
    for new_action in [0, 1, 2]:  # todo notice!
        new_seqs_fp = seqs_fp % new_action
        logging("[READ] read => %s" % new_seqs_fp)
        with open(new_seqs_fp, "rb") as rf:
            seqs = pickle.load(rf)
        seqs = filter_seqs(seqs, user_log_flag, item_log_flag)
        save_pkl("%s/seqs_%d_filter_dream.pkl" % (output_dir, new_action), seqs)
    logging("[END] filter_seqs_dream finished! ")


def split_seqs_copyrec(seqs_fp, output_dir, action):
    print("copy")

    def bigger(t1, t2):
        if dt.strptime(t1, TIME_FORMAT) > dt.strptime(t2, TIME_FORMAT):
            return True
        return False

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logging("[READ] read=> %s" % seqs_fp % action)
    with open(seqs_fp % action, "rb") as rf:
        seqs = pickle.load(rf)
    new_seqs = dict()
    for user_str, user_seq in seqs.items():
        new_user_seq = []
        basket, time_str = user_seq[0]
        new_user_seq.append([basket, time_str, 0, 0])
        for basket, time_str in user_seq[1:]:
            if bigger(PRED_START_TIME, time_str):
                new_user_seq.append([basket, time_str, 1, 0])
            else:
                new_user_seq.append([basket, time_str, 1, 1])
        new_seqs[user_str] = new_user_seq
    save_pkl("%s/seqs_%d_split.pkl" % (output_dir, action), new_seqs)


def sub_enum_seqs(seqs_fp, output_dir, user_enum, item_enum, action=1):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logging("[READ] read => %s" % seqs_fp % action)
    with open(seqs_fp % action, "rb") as rf:
        seqs = pickle.load(rf)
    min_time_dt = None
    max_time_dt = None
    n_logs, n_train_logs, n_test_logs = 0, 0, 0
    n_baskets, n_train_baskets, n_test_baskets = 0, 0, 0
    data_info = dict()
    new_seqs = dict()
    for user_str, user_seq in seqs.items():
        print(user_str, end="\r")
        new_user_seq = []
        user = user_enum[user_str]
        for basket, time_str, is_pred, is_test in user_seq:
            new_basket = []
            for item_str in basket:
                item = item_enum[item_str]
                new_basket.append(item)
            new_user_seq.append([new_basket, time_str, is_pred, is_test])

            n_logs += len(basket)
            if is_test == 1:
                n_test_logs += len(basket)
                n_test_baskets += 1
            else:
                n_train_logs += len(basket)
                n_train_baskets += 1
        n_baskets += len(new_user_seq)
        new_user_seq = sorted(new_user_seq, key=lambda x: dt.strptime(x[1], TIME_FORMAT))
        new_seqs[user] = new_user_seq
        if min_time_dt is None:
            min_time_dt = dt.strptime(new_user_seq[0][1], TIME_FORMAT)
            max_time_dt = dt.strptime(new_user_seq[-1][1], TIME_FORMAT)
        else:
            min_time_dt = min(min_time_dt, dt.strptime(new_user_seq[0][1], TIME_FORMAT))
            max_time_dt = max(max_time_dt, dt.strptime(new_user_seq[-1][1], TIME_FORMAT))
    print(len(new_seqs))
    data_info["n_logs"] = n_logs
    data_info["n_train_logs"] = n_train_logs
    data_info["n_test_logs"] = n_test_logs
    data_info["n_baskets"] = n_baskets
    data_info["n_train_baskets"] = n_train_baskets
    data_info["n_test_baskets"] = n_test_baskets
    data_info["n_users"] = len(user_enum)
    data_info["n_items"] = len(item_enum)
    data_info["from"] = dt.strftime(min_time_dt, TIME_FORMAT)
    data_info["to"] = dt.strftime(max_time_dt, TIME_FORMAT)
    data_info["pred_start"] = str(PRED_START_TIME)
    save_json("%s/data_%d_info.json" % (output_dir, action), data_info, display=True)
    save_pkl("%s/seqs_%d_enum.pkl" % (output_dir, action), new_seqs)


def enum_seqs(seqs_fp, output_dir, action=1):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logging("[READ] read => %s" % seqs_fp)
    with open(seqs_fp % action, "rb") as rf:
        seqs = pickle.load(rf)
    user_enum, item_enum = dict(), dict()
    for user_str, user_seq in seqs.items():
        user_enum.setdefault(user_str, len(user_enum))
        for basket, time_str, is_pred, is_test in user_seq:
            for item_str in basket:
                item_enum.setdefault(item_str, len(item_enum))
    save_pd(item_enum, "item", "old_itemid", "%s/item_enum.csv" % output_dir)
    save_pd(user_enum, "user", "new_userid", "%s/user_enum.csv" % output_dir)

    for new_action in [0, 1, 2, 3]:
        sub_enum_seqs(seqs_fp, output_dir, user_enum, item_enum, new_action)

    logging("[END] enum_seqs finished! ")


def stat_seqs(seqs_fp):
    logging("[READ] read => %s" % seqs_fp)
    with open(seqs_fp, "rb") as rf:
        seqs = pickle.load(rf)
    for user, user_seq in seqs.items():
        n = 0
        basket, time, is_pred, is_test = user_seq[0]
        for basket, time, is_pred, is_test in user_seq:
            if is_test == 1 and is_pred == 0:
                n += 1
        if is_test == 1 and n < 2:
            print(user, n)


def markov_seqs(seqs_fp, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logging("[READ] read => %s" % seqs_fp)
    with open(seqs_fp, "rb") as rf:
        seqs = pickle.load(rf)
    train_data = []
    test_data = []
    for user, user_seq in seqs.items():
        basket_l, time, is_pred, is_test = user_seq[0]
        basket_l = list(map(lambda x: x + 1, basket_l))
        for basket_i, time, is_pred, is_test in user_seq[1:]:
            basket_i = list(map(lambda x: x + 1, basket_i))
            if is_test == 0:
                train_data.append([user + 1, basket_l, basket_i])
            else:
                test_data.append([user + 1, basket_l, basket_i])
            basket_l = basket_i
    save_pkl("%s/markov_train.pkl" % output_dir, train_data)
    save_pkl("%s/markov_test.pkl" % output_dir, test_data)


def get_one_sample(seqs, isTrain=True):
    for user, user_seq in seqs.items():
        x_user = user
        x_baskets = []
        for basket_i, time_str, is_pred, is_test in user_seq:
            if is_pred and isTrain and not is_test:
                y_basket = basket_i
                yield x_user, x_baskets, y_basket
                x_baskets.append(basket_i)
            elif is_test and not isTrain:
                y_basket = basket_i
                yield x_user, x_baskets, y_basket
                x_baskets.append(basket_i)
            elif is_test and isTrain:
                break
            else:
                x_baskets.append(basket_i)


def split_samples(seqs_fp, output_dir):
    # user: [baskets]
    with open(seqs_fp, "rb") as rf:
        seqs = pickle.load(rf)

    train_seqs = dict()
    for user, user_seq in seqs.items():
        new_user_seq = []
        for basket_i, time_str, is_pred, is_test in user_seq:
            if not is_test:
                new_user_seq.append([basket_i, time_str, is_pred, is_test])
        if len(new_user_seq) > 0:
            train_seqs[user] = new_user_seq
    save_pkl("%s/train_seqs.pkl" % output_dir, train_seqs)

    train_input = []
    train_true = []
    for x_user, x_baskets, y_basket in get_one_sample(seqs, isTrain=True):
        train_input.append([x_user, x_baskets])
        train_true.append(y_basket)
    save_pkl("%s/train_input.pkl" % output_dir, train_input)
    with open("%s/train_true.txt" % output_dir, "w") as wf:
        for y_basket in train_true:
            wf.write(",".join([str(e) for e in y_basket]) + "\n")

    test_input = []
    test_true = []
    for x_user, x_baskets, y_basket in get_one_sample(seqs, isTrain=False):
        test_input.append([x_user, x_baskets])
        test_true.append(y_basket)
    save_pkl("%s/test_input.pkl" % output_dir, test_input)
    with open("%s/test_true.txt" % output_dir, "w") as wf:
        for y_basket in test_true:
            wf.write(",".join([str(e) for e in y_basket]) + "\n")

    print(len(test_true))


if __name__ == '__main__':
    DATA_DIR = "/home1/data/tianchi"
    OUTPUT_DIR = "/home1/data/tianchi"
    DATA_LOGS_FILENAMES = ["tianchi_fresh_comp_train_user0.csv",
                           "tianchi_fresh_comp_train_user1.csv",
                           "tianchi_fresh_comp_train_user2.csv",
                           "tianchi_fresh_comp_train_user3.csv"]
    USER_IDX = 1
    ITEM_IDX = 2
    TIME_IDX = 6
    TYPE_IDX = 3
    TIME_FORMAT = "%Y-%m-%d %H"
    LINE_SEG = ","
    PRED_START_TIME = "2014-12-15 00"
    # concat_logs(DATA_DIR, DATA_LOGS_FILENAMES, OUTPUT_DIR, first_line=True)
    # logs2seqs("%s/%s" % (DATA_DIR, "user_log_format1.csv"), OUTPUT_DIR, first_line=True)
    sub_dir = "dream_2_2"
    filter_seqs_dream(seqs_fp="%s/%s" % (DATA_DIR, "seqs_%d.pkl"),
                      output_dir="%s/%s" % (OUTPUT_DIR, sub_dir),
                      user_limit=2, item_limit=2,
                      action=3)
    for aa in [0, 1, 2, 3]:
        split_seqs_copyrec(seqs_fp="%s/%s/%s" % (DATA_DIR, sub_dir, "seqs_%d_filter_dream.pkl"),
                           output_dir="%s/%s" % (OUTPUT_DIR, sub_dir),
                           action=aa)
    enum_seqs(seqs_fp="%s/%s/%s" % (OUTPUT_DIR, sub_dir, "seqs_%d_split.pkl"),
              output_dir="%s/%s/" % (OUTPUT_DIR, sub_dir), action=3)
    # markov_seqs(seqs_fp="%s/%s/%s" % (OUTPUT_DIR, "dream_10_10", "seqs_2_enum.pkl"),
    #             output_dir="%s/%s/" % (OUTPUT_DIR, "dream_10_10"))
    # split_samples(seqs_fp="%s/%s/%s" % (OUTPUT_DIR, "dream", "seqs_2_enum.pkl"),
    #               output_dir="%s/%s/" % (OUTPUT_DIR, "dream"))
