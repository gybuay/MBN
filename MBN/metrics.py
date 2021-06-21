import numpy as np
def str_metric(metric):
    s = ""
    for k, v in metric.items():
        s += "%s=%.6f, " % (k, v)
    return s


def get_NDCG(y_true, y_pred, topk):
    """
    :param y_true: predict basket, list
    :param y_pred: predict basket, list
    :param topk:
    :return:
    """
    count = 0
    dcg = 0
    for pred in y_pred:
        if count >= topk:
            break
        if pred in y_true:
            dcg += (1) / np.math.log2(count + 1 + 1)
        count += 1
    idcg = 0
    num_real_item = len(y_true)
    num_item = int(min(num_real_item, topk))
    for i in range(num_item):
        idcg += (1) / np.math.log2(i + 1 + 1)
    ndcg = dcg / idcg
    return ndcg


def get_hit(y_true, y_pred, topk):
    """
    :param y_true: predict basket, list
    :param y_pred: predict basket, list
    :param topk:
    :return:
    """
    return len(set(y_true) & set(y_pred[:topk]))


def make_metrics(hits_all, actuals_all, ndcgs_all, topk):
    """
    :param hits_all: list of |{PRED} & {TRUE}|
    :param actuals_all: list of |{TRUE}|
    :param ndcgs_all: list of NDCG
    :param topk:
    :return:
    """
    hit_arr = np.array(hits_all)
    actual_arr = np.array(actuals_all)
    ndcgs_arr = np.array(ndcgs_all)

    def get_f1score_macro():
        r = hit_arr / actual_arr
        p = hit_arr / topk
        den = r + p
        den = np.where(den == 0, den + 1, den)
        return np.mean(2 * r * p / den)

    def get_f1score_micro():
        r = np.sum(hit_arr) / len(hit_arr)
        p = np.sum(hit_arr) / (topk * len(hit_arr))
        return 2 * r * p / max((r + p), 1)

    rets = {
        "recall-micro": np.sum(hit_arr) / np.sum(actual_arr),
        "recall-macro": np.mean(hit_arr / actual_arr),
        "precision": np.sum(hit_arr) / (topk * len(hit_arr)),
        "f1score-micro": get_f1score_micro(),
        "f1score-macro": get_f1score_macro(),
        "ndcg-macro": np.mean(ndcgs_arr)
    }
    return rets
