#!/bin/bash
target_epsilons=(1.0 5.0 10.0 15.0 25.0 30.0 35.0 45.0 50.0 55.0 60.0 65.0 70.0 75.0 85.0 90.0 95.0 100.0)
datasets=("cifar10" "cifar100")
seed_values=(10655 11005 12177 14838 20392 21497 25716 41616 42601 54400)
for dataset in "${datasets[@]}"; do
    for target_epsilon in "${target_epsilons[@]}"; do
        # Iterate over the seed values and run curv_score --seed <seed_value>
        for seed_value in "${seed_values[@]}"; do
            # Run curv_score with the seed value
            echo "python private_model_curve_scorer.py --suffix $seed_value --target_epsilon $target_epsilon --dataset $dataset"
        done
    done
done