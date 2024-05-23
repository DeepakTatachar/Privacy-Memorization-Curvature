'''
Code to calculate the loss curvature post training of a DP trained model 
to calculate the correlation between DP and input loss curvature
'''

import os
import multiprocessing


def main():
    import json
    import argparse
    import torch
    import logging
    from utils.str2bool import str2bool
    from utils.inference import inference
    from utils.load_dataset import load_dataset
    from utils.instantiate_model import instantiate_model
    from opacus.validators import ModuleValidator
    from opacus import PrivacyEngine
    import numpy as np
    import random
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   
    parser.add_argument('--dataset',                default='cifar10',      type=str,       help='Set dataset to use')
    parser.add_argument('--gpu',                    default=0,              type=int,       help='Which GPU to use')

    # Dataloader args
    parser.add_argument('--train_batch_size',       default=128,            type=int,       help='Train batch size')
    parser.add_argument('--test_batch_size',        default=256,            type=int,       help='Test batch size')
    parser.add_argument('--val_split',              default=0.0,            type=float,     help='Fraction of training dataset split as validation')
    parser.add_argument('--augment',                default=False,          type=str2bool,  help='Random horizontal flip and random crop')
    parser.add_argument('--padding_crop',           default=4,              type=int,       help='Padding for random crop')
    parser.add_argument('--shuffle',                default=False,          type=str2bool,  help='Shuffle the training dataset')
    parser.add_argument('--random_seed',            default=0,              type=int,       help='Initializing the seed for reproducibility')
    parser.add_argument('--root_path',              default=None,           type=str,       help="Root path for the datasets")
    
    # Model parameters
    parser.add_argument('--save_seed',              default=False,          type=str2bool,  help='Save the seed')
    parser.add_argument('--use_seed',               default=False,          type=str2bool,  help='For Random initialization')
    parser.add_argument('--suffix',                 default='',             type=str,       help='Appended to model name')
    parser.add_argument('--arch',                   default='resnet18',     type=str,       help='Network architecture')
    parser.add_argument("--model_dir",              default=None,           type=str,       help="Where to load models from")

    # Differential Privacy Parameters
    parser.add_argument('--target_epsilon',         default=10.0,           type=float,     help='Target privacy epsilon')
    parser.add_argument('--target_delta',           default=1e-5,           type=float,     help='Target privacy delta')
    parser.add_argument('--max_norm',               default=1.0,            type=float,     help='How much clip grad')

    # Loss Curvature Parameters
    parser.add_argument('--temp',                   default=1.0,            type=float,     help='Temperature Scaling')
    parser.add_argument('--h',                      default=1e-3,           type=float,     help='h for curvature calculation')
    parser.add_argument('--test',                   default=False,          type=str2bool,  help='Calculate curvature on Test Set')
    parser.add_argument("--save_mem_dir",           default=None,           type=str,       help="Where to save scores of curvature to")

    global args
    args = parser.parse_args()

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    version_list = list(map(float, torch.__version__.split(".")[:2]))
    if  version_list[0] <= 1 and version_list[1] < 8: ## pytorch 1.8.0 or below
        torch.set_deterministic(True)
    else:
        torch.use_deterministic_algorithms(True)

    # Create a logger
    logger = logging.getLogger(f'Loss Logger')
    logger.setLevel(logging.INFO)

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

    if not args.save_mem_dir:
        args.save_mem_dir = config['private_precomputed_score_dir']
    

    handler = logging.FileHandler(
        os.path.join(
        log_dir, 
        f'curve_score_dp_{args.dataset}_{args.arch}_eps_{args.target_epsilon}_{args.suffix}.log'), encoding="UTF-8")
    formatter = logging.Formatter(
        fmt=u"%(asctime)s %(levelname)-8s \t %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(args)

    # Parameters
    gpu_id = args.gpu

    # Setup right device to run on
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    dummy_dataset = load_dataset(
        dataset=args.dataset,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        val_split=args.val_split,
        augment=args.augment,
        padding_crop=args.padding_crop,
        shuffle=args.shuffle,
        random_seed=args.random_seed,
        logger=logger,
        root_path=args.root_path)

    dataset_len = dummy_dataset.train_length
    index = np.arange(dataset_len)

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
        index=index,
        root_path=args.root_path)
    
    args.suffix = f"eps_{args.target_epsilon}_{args.suffix}"

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

    net = net.to(device)
    net = ModuleValidator.fix(net)
    ModuleValidator.validate(net, strict=False)

    '''
    Model checkpoint save schema

    saved_training_state = {    
        'model'     : net.state_dict(),
        'dp_accountant': privacy_engine.accountant
    }
    '''
    
    model_path = os.path.join(args.model_dir, args.dataset.lower(), f"{model_name}_and_accountant.ckpt")
    training_state_dict = torch.load(model_path)

    privacy_engine = PrivacyEngine()
    dummy_optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    net, _, _ = privacy_engine.make_private_with_epsilon(
        module=net,
        optimizer=dummy_optimizer,
        data_loader=dataset.train_loader,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        max_grad_norm=args.max_norm,
        epochs=1,
    )

    net.load_state_dict(training_state_dict["model"])
    net = net.to_standard_module()

    test_correct, test_total, test_accuracy = inference(
        net=net,
        data_loader=dataset.test_loader,
        device=device)

    logger.info(
        " Test set: Accuracy: {}/{} ({:.2f}%)".format(
            test_correct,
            test_total,
            test_accuracy))
    
    def get_regularized_curvature_for_batch(batch_data, batch_labels, h=1e-3, niter=10, temp=1):
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

    def score_true_labels_and_save(epoch, test, logger, model_name):
        scores = torch.zeros((dataset_len))
        eig_values = torch.zeros_like(scores)
        labels = torch.zeros_like(scores, dtype=torch.long)
        net.eval()
        total = 0
        dataloader = dataset.train_loader if not test else dataset.test_loader
        for (inputs, targets) in tqdm(dataloader):
            start_idx = total
            stop_idx = total + len(targets)
            idxs = index[start_idx:stop_idx]
            total = stop_idx

            inputs, targets = inputs.cuda(), targets.cuda()
            inputs.requires_grad = True
            net.zero_grad()

            curv_estimate, eig_estimate = get_regularized_curvature_for_batch(inputs, targets, h=args.h, niter=10, temp=args.temp)
            scores[idxs] = curv_estimate.detach().clone().cpu()
            eig_values[idxs] = eig_estimate.detach().clone().cpu()
            labels[idxs] = targets.cpu().detach()

        scores_file_name = f"curv_scores_{epoch}_{model_name}_{args.h}.pt" if not test else f"curv_scores_{epoch}_{model_name}_{args.h}_test.pt"
        eig_file_name = f"eig_values_{epoch}_{model_name}_{args.h}.pt" if not test else f"eig_values_{epoch}_{model_name}_{args.h}_test.pt"
        labels_file_name = f"true_labels{epoch}_{model_name}_{args.h}.pt" if not test else f"true_labels{epoch}_{model_name}_{args.h}_test.pt"

        directory_path = os.path.join(args.save_mem_dir, model_name)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logger.info(f"Created {directory_path}")

        logger.info(f"Saving {scores_file_name}, {eig_file_name}, {labels_file_name}")
        torch.save(scores, os.path.join(args.save_mem_dir, model_name, scores_file_name))
        torch.save(eig_values, os.path.join(args.save_mem_dir, model_name, eig_file_name))
        torch.save(labels, os.path.join(args.save_mem_dir, model_name, labels_file_name))
        return
    
    # Last epoch for dp trained models
    epoch = 20
    logger.info(f'Loading model for epoch {epoch}')
    test_correct, test_total, test_accuracy = inference(net=net, data_loader=dataset.test_loader, device=device)
    logger.info(' Test set: Accuracy: {}/{} ({:.2f}%)'.format(test_correct, test_total, test_accuracy))

    # Calculate curvature score
    score_true_labels_and_save(epoch, args.test, logger, model_name)
       

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()
    
    main()
