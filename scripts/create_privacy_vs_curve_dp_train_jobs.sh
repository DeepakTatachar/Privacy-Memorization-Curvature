# Generate 10 random numbers (0-65536)
# Was generated using -> $(shuf -i 0-65536 -n 7)
random_numbers=(10655 11005 12177 14838 20392 21497 25716 41616 42601 54400)

# Loop over target_epsilons
target_epsilons=(1 5 10 15 25 30 35 45 50 55 60 65 70 75 85 90 95 100)
for target_epsilon in "${target_epsilons[@]}"; do
    # Train python train_dp.py with argument random_seed set to the generated number
    for random_number in "${random_numbers[@]}"; do
        echo "python train_dp.py --epochs 20 --target_epsilon=$target_epsilon --dataset cifar10 --lr 1e-3 --train_batch_size=256 --random_seed=$random_number --suffix=$random_number"
  done
done

# Loop over target_epsilons
for target_epsilon in "${target_epsilons[@]}"; do
    # Train python train_dp.py with argument random_seed set to the generated number
    for random_number in "${random_numbers[@]}"; do
         echo "python train_dp.py --epochs 20 --target_epsilon=$target_epsilon --dataset cifar100 --lr 1e-3 --train_batch_size=256 --random_seed=$random_number --suffix=$random_number"
  done
done