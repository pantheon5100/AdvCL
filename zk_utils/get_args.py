import argparse
from pathlib import Path



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cname', type=str,  default='imagenet_clPretrain',
                        help='')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size')
    # parser.add_argument('--epoch', type=int, default=100,
    #                     help='total epochs')
    parser.add_argument('--save-epoch', type=int, default=100,
                        help='save epochs')

    parser.add_argument('--iter', type=int, default=5,
                        help='The number of iterations for iterative attacks')
    parser.add_argument('--radius', type=int, default=8,
                        help='radius of low freq images')
    parser.add_argument('--ce_weight', type=float, default=0.2,
                        help='cross entp weight')

    
    # parser.add_argument('--checkpoint_dir', type=str,  default='./checkpoint',
    #                     help='')
    parser.add_argument("--checkpoint_dir", default=Path("trained_models"), type=Path)
    
    parser.add_argument("--checkpoint_frequency", default=1, type=int)

    # testing argument
    parser.add_argument('--RA_test_interval', default=5, type=int,
                        help='RA_test_interval')
    parser.add_argument('--test_pgd_epsilon', type=float, default=8,
                        help='att_pgd_epsilon ')
    parser.add_argument('--test_pgd_num_steps', type=int, default=20,
                        help='att_pgd_num_steps ')
    parser.add_argument('--test_pgd_step_size', type=float, default=2,
                        help='att_pgd_step_size ')
    
    
    SUPPORT_TESTBN = ["normal", "pgd"]
    parser.add_argument('--test_BN', default="pgd", type=str, choices=SUPPORT_TESTBN,
                        help='RA_test_interval')

    SUPPORT_LOSSFUNC = ["normal", "pgd"]
    parser.add_argument('--train_loss_func', default=1, type=int,
                        help='train_loss_func')

    # contrastive related
    parser.add_argument('-t', '--nce_t', default=0.5, type=float,
                        help='temperature')
    parser.add_argument('--seed', default=0, type=float,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                            help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    
    # for LinearWarmupCosineAnnealingLR
    parser.add_argument('--min_lr', type=float, default=0.01,
                        help='min_lr ')
    parser.add_argument('--warmup_start_lr', type=float, default=0.01,
                        help='warmup_start_lr')                    
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='warmup_epochs')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='total epochs')
    parser.add_argument('--classifier_lr', type=float, default=0.1,
                        help='classifier_lr ')


    parser.add_argument('--gpus', type=str,  default='4',
                        help='name of the run')
    
    # wandb setting
    parser.add_argument('--name', type=str, default="AdvCL",
                        help='wandb runs name')
    parser.add_argument('--wandb', action='store_true',
                        help='wandb')
    
    # pgd attack related
    parser.add_argument('--att_pgd_epsilon', type=float, default=8,
                        help='att_pgd_epsilon ')
    parser.add_argument('--att_pgd_num_steps', type=int, default=1,
                        help='att_pgd_num_steps ')
    parser.add_argument('--att_pgd_step_size', type=int, default=8,
                        help='att_pgd_step_size ')

    args = parser.parse_args()
    args.decay = args.weight_decay
    args.cosine = True
    args.accelerator = "ddp"

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    elif isinstance(args.gpus, str):
        args.gpus = [int(gpu) for gpu in args.gpus.split(",") if gpu]


    return args

