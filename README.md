## Code Repository for Linking Privacy, Memorization and Input Loss Curvature

This code is the official implementation for the paper, **Unveiling Privacy, Memorization, and Input Curvature Links**. This code contains the experiments for the three links described in the paper.

Setup the environment
---------------------
We recommend using conda as the environment manager. We provide the environment file to run our experiments in `environment.yml` file. Please note this environment needs both PyTorch and TensorFlow installed and this is sometimes causes conflicts. Download the `analysis_checkpoints` directory from [here (link to be updated later)](example.com) and place it in the root folder.

Setup Datasets
--------------
1. Please download the datasets (CIFAR10, CIFAR100 and ImageNet) from respective sources.
2. Please download `imagenet_index.npz` from [Feldman and Zhang](https://github.com/google-research/heldout-influence-estimation) and place it in `build_fz_imagenet/`
3. Use the `build_imagenet.py` in the `build_fz_imagenet` directory to convert to TFRecord dataset.
4. Place the datasets in the following directory structure

```
  +-- data_dir
  |     +-- CIFAR10
  |     |     
  |     +-- CIFAR100
  |     |     
  |     +-- imagenet
```
5. Set the path in `config.json`'s `data_dir` variable. And set the path for TFRecord imagenet dataset in `libdata/indexed_tfrecords.py` line 49 and 51.

Setup Pretrained Models
-----------------------

In our experiments we use pretrained models from [Feldman and Zhang](https://github.com/google-research/heldout-influence-estimation). Please download the models and set the path in  `config.json`'s `fz_model_dir` for both `cifar100` and `imagenet`.

You can also download pretrained private models [here (link to be updated later)](example.com) and set the  `config.json`'s `private_model_dir` variable.

Running Experiments for Link 1 (Memorization and Curvature)
-----------------------------------------------------------
Here we use CIFAR100 and ImageNet dataset. This experiment uses pretrained models from [Feldman and Zhang](https://github.com/google-research/heldout-influence-estimation). Ensure these models are downloaded and paths updated in `config.json`.

**To run CIFAR100 results:**
1. In `config.json` set `fz_precomputed_score_dir`'s `cifar100` path.
2. Run the following command to calculate the curvature scores for cifar100 FZ models
    ```
    python calc_curv_fz_models.py
    ```
3. Run `analyze_cifar100_curv_fz.ipynb` to obtain the results presented in the paper.

**To run ImageNet results:**
1. In `config.json` set `fz_precomputed_score_dir`'s `imagenet` path.
2. Run the following command to calculate the curvature scores for imagenet FZ models. Start idx is the seed to start the curvature calculations from and stop idx is the seed to stop the curvature calculations. Max stop idx is 2000 since FZ models are available for imagenet with 2000 seeds.
    ```
    python calc_curv_fz_imagenet_models.py --start_idx 0 --stop_idx 100
    ```
3. Run `analyze_imagenet_curv_fz.ipynb` to obtain the results presented in the paper.

Running Experiments for Link 2 (Privacy and Curvature)
-----------------------------------------------------------

1. Training private models. To recreate our experiments run 
    ```
    sh scripts/create_privacy_vs_curve_dp_train_jobs.sh > train_privacy_vs_curve_dp.sh
    ```
    to generate the script to run all the training code for all the seeds and all the target privacy budgets

2. Run the training 
    ```
    sh train_privacy_vs_curve_dp.sh
    ```
3. Next step is to compute the curvature scores. Make sure to set variable `private_precomputed_score_dir`'s path for `cifar100` and `cifar10` in `config.json`.
4. Run 
    ```
    sh ./scripts/create_privacy_vs_curve_scorer_job.sh > private_curve_scorer.sh
    ``` 
    to create the script file for running all the curvature calculations for all the seeds, target epsilon and datasets. 
5. To run the curvature calculations run
    ```
    sh private_curve_scorer.sh
    ```
6. Run `loss_v_priv.py` to get the effect of privacy on convergence loss bound result from the paper.
7. Run `curv_privacy.ipynb` to get the curvature vs privacy results from the paper.

Running Experiments for Link 3 (Memorization and Privacy)
-----------------------------------------------------------

1. Training private models. To recreate our experiments run 
    ```
    sh scripts/create_privacy_vs_curve_dp_train_jobs.sh > train_privacy_vs_memorization_dp.sh
    ```
    to generate the script to run all the training code for all the seeds and all the target privacy budgets

2. Run the training 
    ```
    sh train_privacy_vs_memorization_dp.sh
    ```
3.  Run the command to get the results presented in the paper
    ```
    python calc_private_memorization.py
    ```