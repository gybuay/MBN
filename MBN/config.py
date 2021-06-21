import time
import os

class Config:
    def __init__(self, dataset_name):
        self.home_dir = "/home/oubaoyuan/NBR/CopyRec"

        self.dataset_name = dataset_name
        if self.dataset_name == "tianchi":
            self.file_dir = "%s/data/tianchi/dream_2_2/" % self.home_dir
        else:
            self.file_dir = "%s/data/ijcai15/copy/" % self.home_dir
        self.target_task = {"ijcai": 2, "tianchi": 3}[self.dataset_name]
        # tianch i: {"click": 0, "cart": 1, "favor": 2 "buy": 3}
        # ijcai: {"click": 0, "cart": 1, "favor": 3 "buy": 2}
        self.have_tasks = [0, 1, 2, 3]
        self.d_item = 100
        self.d_rnn = 100
        self.d_attn = 100
        self.pool_mode = ["ave", "max", "attn", "sum"][0]
        self.rnn_mode = ["rnn", "gru", "lstm"][0]
        self.loss_mode = ["softmax", "wmse"][0]
        self.n_users = {"ijcai": 30737 + 1, "tianchi": 9197 + 1}[self.dataset_name]
        self.n_items = {"ijcai": 15093 + 1, "tianchi": 17611 + 1}[self.dataset_name]
        self.n_tasks = len(self.have_tasks)
        self.temp_mode = False
        self.cross_mode = True
        self.repeat = 0
        self.mix_mode = 0
        self.epochs = {"ijcai": 20, "tianchi": 60}[self.dataset_name]
        self.display_start = {"ijcai": 0, "tianchi": 0}[self.dataset_name]
        self.display_step = {"ijcai": 1, "tianchi": 1}[self.dataset_name]
        self.batch_size = {"ijcai": 20, "tianchi": 20}[self.dataset_name]
        self.pos_weight = {"ijcai": 10, "tianchi": 10}[self.dataset_name]
        self.neg_weight = {"ijcai": 0.1, "tianchi": 0.01}[self.dataset_name]
        self.lr = {"ijcai": 0.001, "tianchi": 0.001}[self.dataset_name]
        self.reg = 0
        self.sigma = 1
        self.topk = 80
        self.topk_list = [5, 10, 20, 40, 60, 80]
        self.max_to_keep = 3
        self.history = None
        self.n_neg = 1
        self.freq = None
        self.meta_rnn = 1
        self.dropout = 0.2
        self.use_drop = 0
        self.behavior_type = 0
        self.residual = 0
        self.cuda = 0
        self.log_dir = "%s/log/torchmodel/mtlcopy_softmax/%s/copy_%s" % (
            self.home_dir, self.dataset_name,
            time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        os.system("mkdir -p %s" % self.log_dir)
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

    def __str__(self):
        ret = "*" * 60 + "\n"
        for k, v in self.__dict__.items():
            ret += "{}: {}\n".format(k, v)
        ret += "*" * 60 + "\n"
        return ret