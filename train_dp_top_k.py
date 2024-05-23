"""
Code to train deep learning vision classification models using differential privacy using DP-SGD.
This is specifically to train models to calculate memorization.
"""

import os
import multiprocessing

def main():
    import json
    import torch
    import random
    import logging
    import argparse
    import numpy as np
    from opacus import PrivacyEngine
    from utils.str2bool import str2bool
    from utils.inference import inference
    from utils.load_dataset import load_dataset
    from utils.averagemeter import AverageMeter
    from opacus.validators import ModuleValidator
    from utils.instantiate_model import instantiate_model
    from opacus.utils.batch_memory_manager import BatchMemoryManager

    parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument('--epochs',                 default=20,              type=int,       help='Set number of epochs')
    parser.add_argument('--dataset',                default='cifar100',      type=str,       help='Set dataset to use')
    parser.add_argument('--lr',                     default=0.001,           type=float,     help='Learning Rate')
    parser.add_argument('--test_accuracy_display',  default=True,           type=str2bool,  help='Test after each epoch')
    parser.add_argument('--optimizer',              default='SGD',          type=str,       help='Optimizer for training')
    parser.add_argument('--loss',                   default='crossentropy', type=str,       help='Loss function for training')
    parser.add_argument('--resume',                 default=False,          type=str2bool,  help='Resume training from a saved checkpoint')
    parser.add_argument('--gpu',                    default=0,              type=int,       help='Which GPU to use')

    # Dataloader args
    parser.add_argument('--train_batch_size',       default=128,            type=int,       help='Train batch size')
    parser.add_argument('--test_batch_size',        default=256,            type=int,       help='Test batch size')
    parser.add_argument('--val_split',              default=0.0,            type=float,     help='Fraction of training dataset split as validation')
    parser.add_argument('--augment',                default=False,          type=str2bool,  help='Random horizontal flip and random crop')
    parser.add_argument('--padding_crop',           default=4,              type=int,       help='Padding for random crop')
    parser.add_argument('--shuffle',                default=True,           type=str2bool,  help='Shuffle the training dataset')
    parser.add_argument('--random_seed',            default=0,              type=int,       help='Initializing the seed for reproducibility')
    parser.add_argument('--root_path',              default=None,           type=str,       help="Root path for the datasets")
    parser.add_argument("--model_dir",              default=None,           type=str,       help="Where to load models from")

    # Model parameters
    parser.add_argument('--save_seed',              default=False,          type=str2bool,  help='Save the seed')
    parser.add_argument('--use_seed',               default=False,          type=str2bool,  help='For Random initialization')
    parser.add_argument('--suffix',                 default='',             type=str,       help='Appended to model name')
    parser.add_argument('--arch',                   default='resnet18',     type=str,       help='Network architecture')

    # Differential Privacy Parameters
    parser.add_argument('--target_epsilon',         default=20,             type=float,     help='Target privacy epsilon')
    parser.add_argument('--target_delta',           default=1e-5,           type=float,     help='Target privacy delta')
    parser.add_argument('--max_norm',               default=1.0,            type=float,     help='How much clip grad')

    global args
    args = parser.parse_args()

    DELTA = args.target_delta
    MAX_PHYSICAL_BATCH_SIZE = 64

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

    # Specify the path to the config JSON file
    json_file_path = 'config.json'

    # Open and read the JSON file
    with open(json_file_path, 'r') as json_file:
        # Load the JSON data into a Python dictionary
        config = json.load(json_file)

    # Path to log directory
    log_dir = config['log_dir']
    seeds_dir = config['seeds_dir']

    if not args.root_path:
        args.root_path = config['data_dir']

    if not args.model_dir:
        args.model_dir = config['private_model_dir']

    # Create a logger
    logger = logging.getLogger(f'DP Train Logger')
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(
        os.path.join(
        log_dir, 
        f'train_dp_mem_seed_{args.random_seed}_{args.dataset}_{args.arch}_eps_{args.target_epsilon}_{args.suffix}.log'), encoding="UTF-8")
    formatter = logging.Formatter(
        fmt=u"%(asctime)s %(levelname)-8s \t %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(args)

    # Parameters
    num_epochs = args.epochs
    learning_rate = args.lr
    gpu_id = args.gpu

    # Setup right device to run on
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    idxs = np.load(os.path.join(seeds_dir, f"idxs_for_{args.random_seed}.npy"))

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
    
    args.suffix = f"seed_{args.random_seed}_eps_{args.target_epsilon}_{args.suffix}"

    # Instantiate model 
    net, model_name = instantiate_model(
        dataset=dataset,
        arch=args.arch,
        suffix=args.suffix,
        load=args.resume,
        torch_weights=False,
        device=device,
        model_args={},
        logger=logger)

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    if args.resume:
        start_epoch = saved_training_state['epoch']
        optimizer.load_state_dict(saved_training_state['optimizer'])
        net.load_state_dict(saved_training_state['model'])
        best_val_accuracy = saved_training_state['best_val_accuracy']
        best_val_loss = saved_training_state['best_val_loss']
    else:
        start_epoch = 0
        best_val_accuracy = 0.0
        best_val_loss = float('inf')

    net = net.to(device)
    net.train()

    net = ModuleValidator.fix(net)
    ModuleValidator.validate(net, strict=False)
    privacy_engine = PrivacyEngine()

    for param in net.parameters():
        param.requires_grad = True

    # Optimizer
    optimizer = torch.optim.RMSprop(
        net.parameters(),
        lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.6*args.epochs), int(0.8*args.epochs)],
        gamma=0.1)

    net, optimizer, dataset.train_loader = privacy_engine.make_private_with_epsilon(
        module=net,
        optimizer=optimizer,
        data_loader=dataset.train_loader,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        max_grad_norm=args.max_norm,
        epochs=args.epochs,
    )

    # Train model
    for epoch in range(start_epoch, num_epochs, 1):
        net.train()
        train_correct = 0.0
        train_total = 0.0
        save = False
        losses = AverageMeter('Loss', ':.4e')
        logger.info('')
        with BatchMemoryManager(
            data_loader=dataset.train_loader, 
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
            optimizer=optimizer
        ) as memory_safe_data_loader:
            for batch_idx, (data, labels) in enumerate(memory_safe_data_loader):
                data = data.to(device)
                labels = labels.to(device)
                
                # Clears gradients of all the parameter tensors
                optimizer.zero_grad()
                out = net(data)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                losses.update(loss.item())

                with torch.no_grad():
                    train_correct += (out.max(-1)[1] == labels).sum().long().item()
                    train_total += labels.shape[0]

                if (batch_idx + 1) % 100 == 0:
                    curr_acc = 100. * train_correct / train_total
                    epsilon = privacy_engine.get_epsilon(DELTA)
                    logger.info(
                        f"Train Epoch: {epoch} \t"
                        f"Loss: {losses.avg:.6f} "
                        f"Acc@1: {curr_acc:.6f} "
                        f"(ε = {epsilon:.2f}, δ = {DELTA})"
                    )
        
        train_accuracy = float(train_correct) * 100.0 / float(train_total)
        logger.info(
            'Train Epoch: {} Accuracy : {}/{} [ {:.2f}%)]\tLoss: {:.6f}'.format(
                epoch,
                train_correct,
                train_total,
                train_accuracy,
                losses.avg))
       
        # Step the scheduler by 1 after each epoch
        scheduler.step()
        
        val_correct, val_total, val_accuracy, val_loss = -1, -1, -1, -1
        val_accuracy= float('inf')
        save = True

        def save_model_with_dp_accountant(model, accountant, args, model_name, logger):
            '''
            Save the model with the corresponding privacy accountant to be able to validate the privacy after training
            '''
            save_dict = {
                'model': model.state_dict(),
                'dp_accountant': accountant
            }

            save_dir = os.path.join(args.model_dir, args.dataset.lower())
            if not os.path.exists(save_dir):
                logger.info(f"Making dir {save_dir}")
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, model_name + '_and_accountant.ckpt')
            logger.info(f"Saving model @ {save_path}")
            torch.save(save_dict, save_path)

        saved_training_state = {    
            'epoch'     : epoch + 1,
            'optimizer' : optimizer.state_dict(),
            'model'     : net.state_dict(),
            'dp_accountant': privacy_engine.accountant,
            'best_val_accuracy' : best_val_accuracy,
            'best_val_loss' : best_val_loss }

        save_dir = os.path.join(args.model_dir, args.dataset.lower(), 'temp')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, model_name + '.temp')
        torch.save(saved_training_state, save_path)
        
        if save:
            logger.info("Saving checkpoint...")
            save_model_with_dp_accountant(net, privacy_engine.accountant, args, model_name, logger)
            if args.test_accuracy_display:
                # Test model
                # Set the model to eval mode
                test_correct, test_total, test_accuracy = inference(
                    net=net,
                    data_loader=dataset.test_loader,
                    device=device)

                logger.info(
                    " Training set accuracy: {}/{}({:.2f}%) \n" 
                    " Validation set accuracy: {}/{}({:.2f}%)\n"
                    " Test set: Accuracy: {}/{} ({:.2f}%)".format(
                        train_correct,
                        train_total,
                        train_accuracy,
                        val_correct,
                        val_total,
                        val_accuracy,
                        test_correct,
                        test_total,
                        test_accuracy))

    logger.info("End of training without reusing Validation set")
       

if __name__ == "__main__":
    if os.name == 'nt':
        # On Windows calling this function is necessary for multiprocessing
        multiprocessing.freeze_support()
    
    main()

