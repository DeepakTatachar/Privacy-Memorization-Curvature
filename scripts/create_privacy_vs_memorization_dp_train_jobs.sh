for epsilon in 1 10 20 30 40 50; do
    for seed in $(seq 0 39);do
        echo "python train_dp_top_k.py --suffix mem_lr_0.001 --train_batch_size 256 --lr 0.001 --random_seed $seed --target_epsilon $epsilon"
    done
done