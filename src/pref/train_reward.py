import numpy as np
# def start_pref_interface(seg_pipe, pref_pipe, max_segs, synthetic_prefs,
#                          log_dir = ''):
#     def f():
#         # The preference interface needs to get input from stdin. stdin is
#         # automatically closed at the beginning of child processes in Python,
#         # so this is a bit of a hack, but it seems to be fine.
#         # sys.stdin = Vos.fdopen(0)
#         pi.run(seg_pipe=seg_pipe, pref_pipe=pref_pipe)

#     # Needs to be done in the main process because does GUI setup work
#     prefs_log_dir = os.path.join(log_dir, 'pref_interface')
#     pi = PrefInterface(synthetic_prefs=synthetic_prefs,
#                        max_segs=max_segs,
#                        log_dir=prefs_log_dir)
#     proc = Process(target=f, daemon=True)
#     proc.start()
#     return pi, proc
from pref.pref_db import PrefBuffer, PrefDB
from pref.rewardModel import RewardModel
from torch.utils.data import DataLoader


CHECKPOINT_RW = 'src/pref/checkpoints/'
PREFLOGDIR = 'src/pref/preferences/'
def start_reward_predictor_training(cluster_dict,
                                    make_reward_predictor,
                                    load_ckpt_dir,
                                    val_interval,
                                    ckpt_interval = 100):
   
    rew_pred = make_reward_predictor('train', cluster_dict)
    rew_pred.init_network(load_ckpt_dir)
    pref_db_train = PrefDB(maxlen = 5)
    pref_db_val = PrefDB(maxlen = 5)
    pref_db_train.load(PREFLOGDIR + '5'+ '.pkl.gz')
    pref_db_val.load(PREFLOGDIR + '5'+ '.pkl.gz')
    #TODO: Construct reward model and dataloader
    # 
    #FIXME:
    rewardModel = RewardModel(model_arch='simple_cnn', )
    # train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
    #                          num_workers=train_cfg["num_workers"])
    i = 0
    while True:
        rew_pred.train(pref_db_train, pref_db_val, val_interval)
        if i and i % ckpt_interval == 0:
            rew_pred.save()

    # for 

def batch_iter(data, batch_size, shuffle=False):
    idxs = list(range(len(data)))
    if shuffle:
        np.random.shuffle(idxs)  # in-place

    start_idx = 0
    end_idx = 0
    while end_idx < len(data):
        end_idx = start_idx + batch_size
        if end_idx > len(data):
            end_idx = len(data)

        batch_idxs = idxs[start_idx:end_idx]
        batch = []
        for idx in batch_idxs:
            batch.append(data[idx])

        yield batch
        start_idx += batch_size
