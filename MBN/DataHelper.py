from datetime import datetime as dt
import numpy as np
from datetime import datetime

def make_batch_sample(baskets_user, baskets_items, baskets_mask, item_size, copy_size, batch_size):
    """
    :param baskets_user: baskets_user.append(user)
    :param baskets_items: basket_items[task_idx].append((basket_i, list(basket_cum_sum), basket_steps.index(time_str)))
    :param baskets_mask: basket_mask.append(basket_steps.index(time_str))
    :param batch_size:
    :param target_task:
    :return:
    """
    n_history = max([max([tsi
                          for tsi in bm])
                     for bm in baskets_mask]) + 1
    n_tasks = len(baskets_items[0])
    x_item = np.zeros(shape=[n_tasks, batch_size, n_history, item_size], dtype=np.int32)
    x_mask = np.zeros(shape=[batch_size, n_history], dtype=np.int32)
    x_user = np.zeros(shape=[batch_size], dtype=np.int32)
    x_copy = np.zeros(shape=[n_tasks, batch_size, n_history, copy_size], dtype=np.int32)


    for i in range(batch_size):
        for t, basket_task_items in enumerate(baskets_items[i]):
            last_basket_step = 0
            for basket, basket_cs, basket_step in basket_task_items:
                if basket_step >= n_history:
                    break
                
                x_item[t, i, basket_step, :len(basket)] = np.array(basket)

                for mid_basket_step in range(last_basket_step, basket_step - 1):
                    x_copy[t, i, mid_basket_step + 1, :] = x_copy[t, i, last_basket_step]
                x_copy[t, i, basket_step:, :len(basket_cs)] = np.array(basket_cs)
                last_basket_step = basket_step
        for basket_step in baskets_mask[i]:
            x_mask[i, basket_step] = 1
        x_user[i] = baskets_user[i]
    return x_user, x_item, x_mask, x_copy


def batch_generator(task_seqs, target_task, batch_size, have_tasks=[0, 1, 2, 3], train=True, order=False, time_format = "%Y-%m-%d %H"):
    if order:
        cnt, item_size, copy_size = 0, 0, 0
        old_item_size = 0
        old_copy_size = 0
        baskets_user, baskets_items, baskets_mask = [], [], []
        for user in task_seqs[target_task]:
            # ready timestamp
            basket_steps = set()
            for task_idx in have_tasks:
                if user not in task_seqs[task_idx]:
                    continue
                for basket_i, time_str, is_pred, is_test in task_seqs[task_idx][user]:
                    basket_steps.add(time_str)
            basket_steps = list(basket_steps)
            basket_steps = sorted(basket_steps, key=lambda x: dt.strptime(x, time_format))
            # ready ma
            
            basket_mask = []
            for basket_i, time_str, is_pred, is_test in task_seqs[target_task][user]:
                if train and is_test == 0 and is_pred:  # train
                    basket_mask.append(basket_steps.index(time_str))
                elif train and is_test:
                    break
                elif not train and is_pred and is_test:
                    basket_mask.append(basket_steps.index(time_str))
            if len(basket_mask) <= 0:
                continue

            # ready items and copy_size
            basket_items = [[] for _ in have_tasks]
            for ti, task_idx in enumerate(have_tasks):
                if user not in task_seqs[task_idx]:
                    continue
                basket_cum_sum = set()
                user_seq = task_seqs[task_idx][user]
                for basket_i, time_str, is_pred, is_test in user_seq:
                    item_size = max(item_size, len(basket_i))
                    t = basket_steps.index(time_str)
                    if t > basket_mask[-1]:
                        break
                    if train and is_test == 0:  # train
                        basket_cum_sum.update(basket_i)
                        basket_items[ti].append((basket_i, sorted(list(basket_cum_sum)), t))
                    elif train and is_test:
                        break
                    else:
                        basket_cum_sum.update(basket_i)
                        basket_items[ti].append((basket_i, sorted(list(basket_cum_sum)), t))
                copy_size = max(copy_size, len(basket_cum_sum))

            #time delta
            if target_task == 3:
                basket_steps = [datetime.strptime(i, '%Y-%m-%d %H') for i in basket_steps]
            elif target_task == 2:
                basket_steps = [datetime.strptime(i, '%m%d') for i in basket_steps]


            target_time = []
            for tg_time in basket_mask:
                    target_time.append(basket_steps[tg_time])

            for i, tgtime in enumerate(target_time):
                new_basket_items = [[] for _ in have_tasks]
                for t, basket_task_items in enumerate(basket_items):
                    for basket, basket_cs, basket_step in basket_task_items:
                        deltat = tgtime - basket_steps[basket_step]
                        if deltat.days >= 30:
                            continue
                        if deltat.days < 0:
                            break
                        new_basket_items[t].append((basket, basket_cs, basket_step))
                
                new_basket_mask = [basket_steps.index(tgtime)] 

                if len(new_basket_mask) > 0:
                    baskets_user.append(user)
                    baskets_mask.append(new_basket_mask)
                    baskets_items.append(new_basket_items)
                    cnt += 1
                    if cnt == batch_size:
                        for nbi in baskets_items:
                            for t, basket_task_items in enumerate(nbi):
                                for basket, basket_cs, basket_step in basket_task_items:
                                    item_size = max(item_size, len(basket))
                                    copy_size = max(copy_size, len(basket_cs))
                        
                        yield make_batch_sample(baskets_user, baskets_items, baskets_mask,
                                                item_size, copy_size, cnt)
                        baskets_user, baskets_items, baskets_mask = [], [], []
                        cnt, item_size, copy_size = 0, 0, 0
            
            old_item_size = item_size
            old_copy_size = copy_size
            
        if cnt > 0:
            for nbi in baskets_items:
                for t, basket_task_items in enumerate(nbi):
                    for basket, basket_cs, basket_step in basket_task_items:
                        item_size = max(item_size, len(basket))
                        copy_size = max(copy_size, len(basket_cs))
            yield make_batch_sample(baskets_user, baskets_items, baskets_mask,
                                    item_size, copy_size, cnt)
    else:
        raise NotImplementedError()