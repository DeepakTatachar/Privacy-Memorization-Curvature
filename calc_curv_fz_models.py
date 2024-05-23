'''
Description:
    Calculate the CIFAR100 curvature score by converting tf model to pytorch and using 
    the same CIFAR100 order and index as Feldman and Zhang[1].

Reference:
[1] Feldman, V. and Zhang, C. What neural networks memorize and why: Discovering the long tail via influence estimation. 
Advances in Neural Information Processing Systems, 33:2881-2891, 2020.
'''

import os
import torch
import json
import glob
import logging
import argparse
from tqdm import tqdm
import tensorflow as tf
from utils.str2bool import str2bool
from utils.load_dataset import load_dataset
from models.tf_inception import SmallInception
from convert_tf_2_torch import load_checkpoint, copy_tf_2_torch
from models.torch_inception import SmallInception as SmallInceptionTorch

parser = argparse.ArgumentParser(
    description="Calculate the CIFAR100 curvature score by converting tf model to pytorch", 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("--dataset", default="CIFAR100", type=str, help="Set dataset to use")
parser.add_argument("--data_dir", default=None, type=str, help="Where to load data from")
parser.add_argument("--model_dir", default=None, type=str, help="Where to load fz models from")
parser.add_argument("--save_mem_dir", default=None, type=str, help="Where to save scores of curvature to")
parser.add_argument("--gpus", default="0,1,2", type=str, help="Which gpus to use")

# Dataloader args
parser.add_argument("--train_batch_size", default=1024, type=int, help="Train batch size")
parser.add_argument("--test_batch_size", default=1024, type=int, help="Test batch size")
parser.add_argument("--val_split", default=0.0, type=float, help="Fraction of training dataset split as validation")
parser.add_argument("--augment", default=False, type=str2bool, help="Random horizontal flip and random crop")
parser.add_argument("--padding_crop", default=4, type=int, help="Padding for random crop")
parser.add_argument("--shuffle", default=False, type=str2bool, help="Shuffle the training dataset")
parser.add_argument("--random_seed", default=0, type=int, help="Initializing the seed for reproducibility")

# Loss Curvature Parameters
parser.add_argument('--temp', default=1.0, type=float, help='Temperature Scaling')
parser.add_argument('--h', default=1e-3, type=float, help='h for curvature calculation')

global args
args = parser.parse_args()

logger = logging.getLogger("tensorflow")
logger.setLevel(logging.INFO)

# Specify the path to the config JSON file
json_file_path = 'config.json'

# Open and read the JSON file
with open(json_file_path, 'r') as json_file:
    # Load the JSON data into a Python dictionary
    config = json.load(json_file)

# Path to log directory
log_dir = config['log_dir']

if not args.data_dir:
    args.data_dir = config['data_dir']

if not args.save_mem_dir:
    args.save_mem_dir = config['fz_precomputed_score_dir'][args.dataset.lower()]

if not args.model_dir:
    args.model_dir = config['fz_model_dir'][args.dataset.lower()]


# Check if logs directory exists if not create directory
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


gpus = list(map(int, args.gpus.split(",")))
handler = logging.FileHandler(os.path.join(log_dir, f"save_{args.dataset.lower()}_fz_curve.log"))
formatter = logging.Formatter(fmt=f"%(asctime)s %(levelname)-8s %(message)s ", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info(args)
local_rank = 0

dataset = load_dataset(
    dataset=args.dataset,
    train_batch_size=args.train_batch_size,
    test_batch_size=args.test_batch_size,
    val_split=args.val_split,
    augment=args.augment,
    padding_crop=args.padding_crop,
    shuffle=args.shuffle,
    root_path=args.data_dir,
    random_seed=args.random_seed,
    mean=[0, 0, 0],
    std=[1, 1, 1],
    logger=logger,
    workers=10
)

gpus = tf.config.list_physical_devices("GPU")
if len(gpus) > 0:
    log_dev_conf = tf.config.LogicalDeviceConfiguration(memory_limit=200)  # 100 MB

    # Apply the logical device configuration to the first GPU
    tf.config.set_logical_device_configuration(gpus[local_rank], [log_dev_conf])

# Setup right device to run on
device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
net_py = SmallInceptionTorch(num_classes=dataset.num_classes)
net_tf = SmallInception(num_classes=dataset.num_classes)

def get_regularized_curvature_for_batch(net, batch_data, batch_labels, h=1e-3, niter=10, temp=1):
    num_samples = batch_data.shape[0]
    net.eval()
    regr = torch.zeros(num_samples)
    eigs = torch.zeros(num_samples)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(niter):
        v = torch.randint_like(batch_data, high=2).cuda()
        # Generate Rademacher random variables
        for v_i in v:
            v_i[v_i == 0] = -1

        v = h * (v + 1e-7)

        batch_data.requires_grad_()
        outputs_pos = net(batch_data + v)
        outputs_orig = net(batch_data)
        loss_pos = criterion(outputs_pos / temp, batch_labels)
        loss_orig = criterion(outputs_orig / temp, batch_labels)
        grad_diff = torch.autograd.grad((loss_pos-loss_orig), batch_data)[0]

        regr += grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1).cpu().detach()
        eigs += torch.diag(torch.matmul(v.reshape(num_samples,-1), grad_diff.reshape(num_samples,-1).T)).cpu().detach()
        net.zero_grad()
        if batch_data.grad is not None:
            batch_data.grad.zero_()

    eig_values = eigs / niter
    curvature = regr / niter
    return curvature, eig_values

def score_true_labels_and_save(net, dataset_len, dataset, index, save_mem_dir, logger, model_name):
    scores = torch.zeros((dataset_len))
    eig_values = torch.zeros_like(scores)
    labels = torch.zeros_like(scores, dtype=torch.long)
    net.eval()
    total = 0
    dataloader = dataset.train_loader
    if index is None:
        index = list(range(dataset_len))

    for (inputs, targets) in tqdm(dataloader):
        start_idx = total
        stop_idx = total + len(targets)
        idxs = index[start_idx:stop_idx]
        total = stop_idx

        inputs, targets = inputs.cuda(), targets.cuda()
        inputs.requires_grad = True
        net.zero_grad()

        curv_estimate, eig_estimate = get_regularized_curvature_for_batch(
            net, 
            inputs, 
            targets, 
            h=args.h, 
            niter=10, 
            temp=args.temp)

        scores[idxs] = curv_estimate.detach().clone().cpu()
        eig_values[idxs] = eig_estimate.detach().clone().cpu()
        labels[idxs] = targets.cpu().detach()

    scores_file_name = f"curv_scores_{model_name}_{args.h}.pt"
    eig_file_name = f"eig_values_{model_name}_{args.h}.pt"
    labels_file_name = f"true_labels_{model_name}_{args.h}.pt"

    directory_path = os.path.join(save_mem_dir, model_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created {directory_path}")

    logger.info(f"Saving {scores_file_name}, {eig_file_name}, {labels_file_name}")
    torch.save(scores, os.path.join(directory_path, scores_file_name))
    torch.save(eig_values, os.path.join(directory_path, eig_file_name))
    torch.save(labels, os.path.join(directory_path, labels_file_name))
    return

for exp_idx in range(1000):
    for ratio_int in [1,2,3,4,5,6,7,8,9]:
        ratio = f"0.{ratio_int}"
        logger.info("-" * 40)
        logger.info(f"Ratio {ratio}")
        checkpoint_dir = os.path.join(args.model_dir, f"{ratio}", f"{exp_idx}", "checkpoints")
        ckpt_list = glob.glob(os.path.join(checkpoint_dir, "ckpt-*.index"))
        ckpt_path = ckpt_list[0][:-6]
        load_results = load_checkpoint(net_tf, checkpoint_dir)
        copy_tf_2_torch(net_tf, net_py)
        net_py = net_py.eval()
        net_dp = torch.nn.DataParallel(net_py, device_ids=gpus)
        logger.info("-" * 20)
        logger.info(f"Experiment idx {exp_idx}, ratio {ratio}")

        net_py.eval()
        net_py.to(device)
        model_name = f"{args.dataset.lower()}_small_inception_{ratio}_{exp_idx}"
        score_true_labels_and_save(net_dp, dataset.train_length, dataset, None, args.save_mem_dir, logger, model_name)
