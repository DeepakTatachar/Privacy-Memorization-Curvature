'''
Description:

'''

import os
import multiprocessing

def calc_loss_acc_for_dp_models(filename):
    import json
    import torch
    import pickle
    import logging
    import argparse
    from utils.str2bool import str2bool
    from utils.load_dataset import load_dataset
    from utils.averagemeter import AverageMeter
    from opacus.validators import ModuleValidator
    from utils.instantiate_model import instantiate_model

    parser = argparse.ArgumentParser(description='Calculate Memorization Score', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Dataset parameters
    parser.add_argument('--dataset',                default='cifar100',      type=str,      help='Set dataset to use only cifar100 supported')
    parser.add_argument('--loss',                   default='crossentropy', type=str,       help='Loss function for training')

    # Dataloader args
    parser.add_argument('--train_batch_size',       default=128,            type=int,       help='Train batch size')
    parser.add_argument('--test_batch_size',        default=256,            type=int,       help='Test batch size')
    parser.add_argument('--val_split',              default=0.0,            type=float,     help='Fraction of training dataset split as validation')
    parser.add_argument('--augment',                default=False,          type=str2bool,  help='Random horizontal flip and random crop')
    parser.add_argument('--padding_crop',           default=4,              type=int,       help='Padding for random crop')
    parser.add_argument('--shuffle',                default=False,          type=str2bool,  help='Shuffle the training dataset')
    parser.add_argument('--random_seed',            default=0,              type=int,       help='Initializing the seed for reproducibility')
    parser.add_argument('--root_path',              default=None,           type=str,       help="Root path for the datasets")
    parser.add_argument("--model_dir",              default=None,           type=str,       help="Where to load models from")

    # Model parameters
    parser.add_argument('--save_seed',              default=False,          type=str2bool,  help='Save the seed')
    parser.add_argument('--use_seed',               default=False,          type=str2bool,  help='For Random initialization')
    parser.add_argument('--suffix',                 default='',             type=str,       help='Appended to model name')
    parser.add_argument('--arch',                   default='resnet18',     type=str,       help='Network architecture')

    global args
    args = parser.parse_args()

    # Specify the path to the config JSON file
    json_file_path = 'config.json'

    # Open and read the JSON file
    with open(json_file_path, 'r') as json_file:
        # Load the JSON data into a Python dictionary
        config = json.load(json_file)

    # Path to log directory
    log_dir = config['log_dir']

    if not args.root_path:
        args.root_path = config['data_dir']

    if not args.model_dir:
        args.model_dir = config['private_model_dir']

    # Check if logs directory exists if not create directory
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a logger
    logger = logging.getLogger(f'Train Logger')
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(
        os.path.join(
        log_dir, 
        f'mem_v_priv_fixed.log'), encoding="UTF-8")
    formatter = logging.Formatter(
        fmt=u"%(asctime)s %(levelname)-8s \t %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(args)

    # Parameters
    gpu_id = 0

    # Setup right device to run on
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Use the following transform for training and testing
    dataset = load_dataset(
        dataset=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=args.val_split,
        augment=False,
        padding_crop=args.padding_crop,
        shuffle=False,
        random_seed=args.random_seed,
        logger=logger,
        root_path=args.root_path,
        workers=2)
    
    # To mimic the privacy engine wrapper from Opacus since the saved model
    # was stored this way
    class DummyNet(torch.nn.Module):
        def __init__(self, net) -> None:
            super(DummyNet, self).__init__()
            self._module = net
    
    results = {}
    for eps in [1.0, 10.0, 20.0, 30.0, 40.0, 50.0]:
        save_dir = os.path.join(args.model_dir, args.dataset.lower())
        models = os.listdir(save_dir)
        seeds = []
        for model_name in models:
            if args.dataset.lower() in model_name and f"eps_{eps}_mem_lr_0.001" in model_name:
                seeds.append(model_name.split("_")[3])
        for seed in seeds:
            args.suffix = f"eps_{float(eps)}_{seed}"

            # Instantiate model 
            net, model_name = instantiate_model(
                dataset=dataset,
                arch=args.arch,
                suffix=args.suffix,
                load=False,
                torch_weights=False,
                device=device,
                model_args={},
                logger=logger)
                         
            net = ModuleValidator.fix(net)
            ModuleValidator.validate(net, strict=False)

            model_and_acc_path = os.path.join(save_dir, f"{args.dataset.lower()}_resnet18_seed_{seed}_eps_{eps}_mem_lr_0.001_and_accountant.ckpt")
            save_checkpoint = torch.load(model_and_acc_path)
            logger.info(f"Loading from model {model_and_acc_path}")
            dummy_net = DummyNet(net)
            dummy_net.load_state_dict(save_checkpoint['model'])
            net.eval()

            # Loss
            criterion = torch.nn.CrossEntropyLoss()
            losses = AverageMeter('Loss', ':.4e')
            train_correct = 0
            train_total = 0
            probs = []
            with torch.no_grad():
                for data, labels in dataset.train_loader:
                    data = data.to(device)
                    labels = labels.to(device)
                    out = net(data)
                    prob = torch.nn.functional.softmax(out, dim=1)
                    prob = torch.nn.functional.one_hot(labels, dataset.num_classes) * prob
                    prob = prob.sum(1)
                    probs.extend(prob.cpu().numpy().tolist())
                    loss = criterion(out, labels)
                    losses.update(loss.item())
                    train_correct += (out.max(-1)[1] == labels).sum().long().item()
                    train_total += labels.shape[0]
            
            train_accuracy = float(train_correct) * 100.0 / float(train_total)
            logger.info(
                'Train Accuracy : {}/{} = {:.2f}% \tLoss: {:.6f}'.format(
                    train_correct,
                    train_total,
                    train_accuracy,
                    losses.avg
                )
            )

            if eps in results:
                results[eps].append((int(seed), losses.avg, train_accuracy, probs))
            else:
                results[eps] = [(int(seed), losses.avg, train_accuracy, probs)]
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()

    import scienceplots
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    markers = ['x', '+']
    plt.style.use(['ieee', 'science', 'grid'])

    textwidth = 3.31314
    aspect_ratio = 6/8
    scale = 1.0
    width = textwidth * scale
    height = width * aspect_ratio
    fig = plt.figure(figsize=(width, height))
    colors=['mediumblue', 'red']

    filename = './data/dp_mem_fixed_results_lr0.001.pkl'

    if not os.path.exists(filename):
        calc_loss_acc_for_dp_models(filename)

    with open(filename, 'rb') as f:
        results = pickle.load(f)

    top_k = 500
    npz = np.load('./analysis_checkpoints/cifar100/cifar100_infl_matrix.npz', allow_pickle=True)
    fz_scores = pd.DataFrame.from_dict({item: npz[item] for item in ['tr_labels', 'tr_mem']})
    fz_sorted = fz_scores.sort_values(by='tr_mem', ascending=False)[:top_k]
    most_mem = fz_sorted.index.to_numpy()

    mem_score_for_eps = {}
    epsilons = []
    mean_mem_score = []

    for eps, results in results.items():
        prob_mem_all = []
        mask_mem_all = []
        for seed, loss, acc, prob in results:
            idxs = np.load(f"./seeds/idxs_for_{seed}.npy")
            prob = np.array(prob)
            memorized_mask = np.zeros_like(prob)
            memorized_mask[idxs] = 1
            prob_mem_all.append(prob[most_mem])
            mask_mem_all.append(memorized_mask[most_mem])
        
        prob_mem_all = np.array(prob_mem_all)
        mask_mem_all = np.array(mask_mem_all)
        pr1 = (prob_mem_all * mask_mem_all).mean(0)
        pr0 = (prob_mem_all * (1 - mask_mem_all)).mean(0)
        mem_score_for_eps[eps] = pr1 - pr0
        epsilons.append(eps)
        mean_mem_score.append(np.abs(mem_score_for_eps[eps]).mean())
 
    def curv_func(x):
        return 1 - np.exp(-x)

    plt.style.use(['science','grid'])


    plt.scatter(epsilons, mean_mem_score, marker=markers[0], c=colors[0], alpha=0.7, label='Empirical Memorization Score')
    legend = plt.legend(fancybox=False, edgecolor="black")
    legend.get_frame().set_linewidth(0.5)


    lin_model = LinearRegression()
    tr_x = np.array(epsilons)
    tr_y = np.array(mean_mem_score)
    plotx = np.linspace(0, 50, 100)
    plt.semilogy(plotx, curv_func(plotx), c=colors[1], label='Theoretical Bound', linestyle='dashed', lw=0.6)
    plt.ylim(bottom = 1e-4)
    plt.legend(loc='lower right')
    plt.xlabel("Privacy Budget")
    plt.ylabel("Memorization")
    plt.savefig("./plots/priv_mem.png", dpi=300)
