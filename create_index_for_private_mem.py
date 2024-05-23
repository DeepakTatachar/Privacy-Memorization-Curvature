import numpy as np
import  pandas as pd

top_k = 500
npz = np.load('./analysis_checkpoints/cifar100/cifar100_infl_matrix.npz', allow_pickle=True)
fz_scores = pd.DataFrame.from_dict({item: npz[item] for item in ['tr_labels', 'tr_mem']})
fz_sorted = fz_scores.sort_values(by='tr_mem', ascending=False)[:top_k]
most_mem = fz_sorted.index.to_numpy()
ratio = 0.7

for seed_idx in range(100):
    all_idx = np.arange(0, 50000, 1)
    not_most_mem = np.setdiff1d(all_idx, most_mem)
    np.random.shuffle(not_most_mem)

    not_most_mem = not_most_mem[:int(ratio * len(not_most_mem))]
    most_mem_clone = np.array(most_mem)
    np.random.shuffle(most_mem_clone)

    # Randomly choose 50% of the top most memorized
    most_mem_clone = most_mem_clone[:top_k//2]
    idxs_for_seed = np.concatenate([not_most_mem, most_mem_clone])
    np.save(f"./seeds/idxs_for_{seed_idx}", idxs_for_seed)