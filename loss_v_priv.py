'''
Description:
    Empirical loss bound calculation for private models. 
    We obtain the empirical values and also fit the theory model.
    The resulting plots are saved.
'''

import os
import multiprocessing

def calc_loss_acc_for_dp_models():
    import json
    import torch
    import pickle
    import logging
    import argparse
    import numpy as np
    from utils.str2bool import str2bool
    from utils.load_dataset import load_dataset
    from utils.averagemeter import AverageMeter
    from opacus.validators import ModuleValidator
    from utils.instantiate_model import instantiate_model

    parser = argparse.ArgumentParser(description='Plot loss vs privacy', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Dataset parameters
    parser.add_argument('--dataset',                default='cifar100',     type=str,       help='Set dataset to use')
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

    # Create a logger
    logger = logging.getLogger(f'Loss vs Privacy')
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(
        os.path.join(
        log_dir, 
        f'loss_v_priv.log'), encoding="UTF-8")
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
    idxs = np.load(f"./seeds/idxs_for_{args.random_seed}.npy")

    # Use the following transform for training and testing
    dataset = load_dataset(
        dataset=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=args.val_split,
        augment=args.augment,
        padding_crop=args.padding_crop,
        shuffle=args.shuffle,
        random_seed=args.random_seed,
        logger=logger,
        index=idxs,
        root_path=args.root_path,
        workers=0)
    
    class DummyNet(torch.nn.Module):
        def __init__(self, net) -> None:
            super(DummyNet, self).__init__()
            self._module = net
    
    results = {}
    for eps in np.arange(5, 105, 5).tolist():
        save_dir = os.path.join(args.model_dir, args.dataset.lower())
        models = os.listdir(save_dir)
        seeds = []
        for model_name in models:
            if args.dataset.lower() in model_name and f"eps_{eps}.0" in model_name:
                seeds.append(model_name.split("_")[4])
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

            model_and_acc_path = os.path.join(save_dir, model_name + '_and_accountant.ckpt')
            save_checkpoint = torch.load(model_and_acc_path)
            logger.info(f"Loading from model {model_and_acc_path}")
            dummy_net = DummyNet(net)
            dummy_net.load_state_dict(save_checkpoint['model'])
            net.eval()

            # Loss
            criterion = torch.nn.CrossEntropyLoss()
            losses = AverageMeter('Loss', ':.4e')
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for data, labels in dataset.test_loader:
                    data = data.to(device)
                    labels = labels.to(device)
                    out = net(data)
                    loss = criterion(out, labels)
                    losses.update(loss.item())
                    test_correct += (out.max(-1)[1] == labels).sum().long().item()
                    test_total += labels.shape[0]
            
            test_accuracy = float(test_correct) * 100.0 / float(test_total)
            logger.info(
                'Test Accuracy : {}/{} = {:.2f}% \tLoss: {:.6f}'.format(
                    test_correct,
                    test_total,
                    test_accuracy,
                    losses.avg
                )
            )

            if eps in results:
                results[eps].append((int(seed), losses.avg, test_accuracy))
            else:
                results[eps] = [(int(seed), losses.avg, test_accuracy)]
    
    with open('./data/dp_loss_results.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()

    import scienceplots
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from scipy.optimize import curve_fit

    markers = ['x', '+']
    plt.style.use(['ieee', 'science', 'grid'])

    textwidth = 3.31314
    aspect_ratio = 6/8
    scale = 1.0
    width = textwidth * scale
    height = width * aspect_ratio
    fig = plt.figure(figsize=(width, height))
    colors=['mediumblue', 'red']

    def curv_func(x, a, b, c):
        return a + b * np.exp(-c * x)

    if not os.path.exists('./data/dp_loss_results.pkl'):
        calc_loss_acc_for_dp_models()
    else:
        with open('./data/dp_loss_results.pkl', 'rb') as f:
            results = pickle.load(f)

        plt.style.use(['science','grid'])

        eps_data = []
        loss_data = []
        for eps, data in results.items():
            eps_data.append(float(eps))
            loss_data.append(np.array([d[1] for d in data]).mean())

        plt.scatter(eps_data, loss_data, marker=markers[0], c=colors[0], alpha=0.7, label='Empirical Loss Bound')
        legend = plt.legend(fancybox=False, edgecolor="black")
        legend.get_frame().set_linewidth(0.5)


        lin_model = LinearRegression()
        tr_x = np.array(eps_data)
        tr_y = np.array(loss_data)
        popt, pcov = curve_fit(curv_func, tr_x, tr_y)
        plotx = np.linspace(1, 100, 100)
        plt.plot(plotx, curv_func(plotx, *popt), c=colors[1], label='Theory Best Fit', linestyle='dashed', lw=0.6)
        plt.legend()
        plt.xlabel("Privacy Budget")
        plt.ylabel("Loss")
        
        plt.title('Loss vs Privacy')
        plt.tight_layout()
        plt.savefig("./plots/eps_v_loss.png", dpi=300)

        