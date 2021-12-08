from __future__ import print_function

import argparse
import numpy as np
import os, csv
from dataset import CIFAR10IndexPseudoLabelEnsemble
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.backends.cudnn as cudnn

from utils import progress_bar, TwoCropTransformAdv
from losses import SupConLoss, OurLoss1, OurLoss2
from models.resnet_cifar_multibn_ensembleFC import resnet18 as ResNet18
import random
from fr_util import generate_high
from utils import adjust_learning_rate, warmup_learning_rate


import os
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger


def static_lr(
    get_lr, param_group_indexes, lrs_to_replace
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs

def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k
):
    """Computes the accuracy over the k top predictions for the specified values of k.
    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).
    Returns:
        Sequence[int]:  accuracies at the desired k.
    """

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def weighted_mean(outputs, key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.
    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.
    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, 
    learning_rate, 
    decay, 
    nce_t, 
    ce_weight, 
    classifier_lr, 
    epsilon, 
    iter,
    min_lr,
    warmup_start_lr,
    warmup_epochs,
    radius,
    max_epochs,
    **kwargs,
    ):
        super().__init__()
        # self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        # self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))


        self.learning_rate = learning_rate
        self.decay = decay
        self.max_epochs = max_epochs
        self.ce_weight = ce_weight
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.classifier_lr = classifier_lr
        self.radius = radius

        # config = {
        #     'epsilon': epsilon / 255.,
        #     'num_steps': iter,
        #     'step_size': 2.0 / 255,
        #     'random_start': True,
        #     'loss_func': 'xent',
        # }

        config = {
            'epsilon': epsilon / 255.,
            'num_steps': 1,
            'step_size': 8.0 / 255,
            'random_start': True,
            'loss_func': 'xent',
        }

        bn_names = ['normal', 'pgd', 'pgd_ce']

        self.model = ResNet18(bn_names=bn_names)
        self.net = AttackPGD(self.model, config, ce_weight, radius)

        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.contrast_criterion = SupConLoss(temperature=nce_t)
        # self.contrast_criterion = OurLoss1(temperature=nce_t)
        # self.contrast_criterion = OurLoss2(temperature=nce_t)



        self.classifier = nn.Linear(512, 10)


    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding
    
    def base_forward(self, X: torch.Tensor):
        """Basic forward that allows children classes to override forward().
        Args:
            X (torch.Tensor): batch of images in tensor format.
        Returns:
            Dict: dict of logits and features.
        """

        # feats = self.model(X, bn_name='normal', return_feat=True)
        feats = self.model(X, bn_name='pgd', return_feat=True)

        logits = self.classifier(feats.detach())
        return {
            "logits": logits,
            "feats": feats,
        }

    def _base_shared_step(self, X: torch.Tensor, targets: torch.Tensor):
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.
        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.
        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        out = self.base_forward(X)
        logits = out["logits"]

        loss = F.cross_entropy(logits, targets, ignore_index=-1)
        # handle when the number of classes is smaller than 5
        top_k_max = min(5, logits.size(1))
        acc1, acc5 = accuracy_at_k(logits, targets, top_k=(1, top_k_max))

        return {**out, "loss": loss, "acc1": acc1, "acc5": acc5}
    

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        # x, y = batch
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        # self.log("train_loss", loss)

        # adjust_learning_rate(args, optimizer, epoch+1)


        inputs, true_label, targets, ind = batch
        # import ipdb; ipdb.set_trace()

        # tt = []
        # for tt_ in targets:
        #     tt
        # targets = tt
        image_t1, image_t2, image_org = inputs

        # warmup_learning_rate(args, epoch+1, batch_idx, len(train_loader), optimizer)
        # attack contrast
        # optimizer.zero_grad()
        # import ipdb; ipdb.set_trace()
        x1, x2, x_cl, x_ce, x_HFC = self.net(image_t1, image_t2, image_org, targets, self.contrast_criterion)
        f_proj, f_pred = self.model(x_cl, bn_name='pgd', contrast=True)
        # fce_proj, fce_pred, logits_ce = self.model(x_ce, bn_name='pgd_ce', contrast=True, CF=True, return_logits=True, nonlinear=False)
        f1_proj, f1_pred = self.model(x1, bn_name='normal', contrast=True)
        f2_proj, f2_pred = self.model(x2, bn_name='normal', contrast=True)
        # # f_high_proj, f_high_pred = self.model(x_HFC, bn_name='normal', contrast=True)
        # features = torch.cat(
        #         [f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_high_proj.unsqueeze(1)], dim=1)
        features = torch.cat(
                [f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1)], dim=1)
        contrast_loss = self.contrast_criterion(features)

        contrast_loss = simclr_anchor_loss_func(f_proj, f1_proj) 
        contrast_loss += simclr_anchor_loss_func(f1_proj, f2_proj) 
        contrast_loss += simclr_anchor_loss_func(f2_proj, f1_proj) 



        # simclr_anchor_loss_func
        # ce_loss = 0
        # for label_idx in range(5):
        #     tgt = targets[label_idx].long()
        #     lgt = logits_ce[label_idx]
        #     ce_loss += self.ce_criterion(lgt, tgt) / 5.

        # loss = contrast_loss + ce_loss * self.ce_weight
        loss = contrast_loss

        

        outs = self._base_shared_step(image_org, true_label)

        class_loss = outs["loss"]

        metrics = {
            "train_class_loss": outs["loss"],
            "train_acc1": outs["acc1"],
            "train_acc5": outs["acc5"],
            "ad_ssl_loss": loss
        }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        return loss + class_loss


    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: int = None
    ):
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.
        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.
        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), the classification loss
                and accuracies.
        """

        X, targets = batch
        batch_size = targets.size(0)

        out = self._base_shared_step(X, targets)

        # robustness accuracy testing
        torch.set_grad_enabled(True)
        adv_images = self.PGD_val_adv(X, targets)
        torch.set_grad_enabled(False)
        out_ra = self._base_shared_step(adv_images, targets)

        metrics = {
            "batch_size": batch_size,
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],
            "RA_val_loss": out_ra["loss"],
            "RA_val_acc1": out_ra["acc1"],
            "RA_val_acc5": out_ra["acc5"],
        }
        return metrics

    def validation_epoch_end(self, outs):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        val_acc1 = weighted_mean(outs, "val_acc1", "batch_size")
        val_acc5 = weighted_mean(outs, "val_acc5", "batch_size")
        ra_val_loss = weighted_mean(outs, "RA_val_loss", "batch_size")
        ra_val_acc1 = weighted_mean(outs, "RA_val_acc1", "batch_size")
        ra_val_acc5 = weighted_mean(outs, "RA_val_acc5", "batch_size")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5,
        "RA_val_loss": ra_val_loss, "RA_val_acc1": ra_val_acc1, "RA_val_acc5": ra_val_acc5}

        self.log_dict(log, sync_dist=True)

    def PGD_val_adv(self, images, labels):
        r"""
        Overridden.
        """
        # images = images.clone().detach().to(self.device)
        # labels = labels.clone().detach().to(self.device)


        loss = nn.CrossEntropyLoss()

        steps = 5
        alpha = 2.0 / 255
        eps = 8.0 / 255

        adv_images = images.clone().detach()

        # if self.random_start:
            # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()



        for _ in range(steps):
            adv_images.requires_grad = True

            # outputs = self.model(adv_images, bn_name='normal', return_feat=True)
            outputs = self.model(adv_images, bn_name='pgd', return_feat=True)

            outputs = self.classifier(outputs)

            # Calculate loss
            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    @property
    def learnable_params(self):
        """Defines learnable parameters for the base class.
        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            # {"name": "backbone", "params": self.model.parameters()},
            {"name": "pgd", "params": self.net.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            },
        ]

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # collect learnable parameters
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        optimizer = torch.optim.SGD(self.learnable_params, lr=self.learning_rate, momentum=0.9, weight_decay=self.decay)

        scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.warmup_epochs,
                max_epochs=self.max_epochs,
                warmup_start_lr=self.warmup_start_lr,
                eta_min=self.min_lr,
            )

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr,
                get_lr=scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            )
            scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]



def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str,  default='advcl_cifar10',
                        help='name of the run')
    parser.add_argument('--cname', type=str,  default='imagenet_clPretrain',
                        help='')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--epoch', type=int, default=100,
                        help='total epochs')
    parser.add_argument('--save-epoch', type=int, default=100,
                        help='save epochs')
    parser.add_argument('--epsilon', type=float, default=8,
                        help='The upper bound change of L-inf norm on input pixels')
    parser.add_argument('--iter', type=int, default=5,
                        help='The number of iterations for iterative attacks')
    parser.add_argument('--radius', type=int, default=8,
                        help='radius of low freq images')
    parser.add_argument('--ce_weight', type=float, default=0.2,
                        help='cross entp weight')

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
                        help='max_epochs')
    parser.add_argument('--classifier_lr', type=float, default=0.1,
                        help='classifier_lr ')


    parser.add_argument('--gpus', type=str,  default='0',
                        help='name of the run')
    parser.add_argument('--wandb', action='store_true',
                        help='wandb')

    args = parser.parse_args()
    args.epochs = args.epoch
    args.decay = args.weight_decay
    args.cosine = True
    args.accelerator = "ddp"

    if isinstance(args.gpus, int):
        args.gpus = [args.gpus]
    elif isinstance(args.gpus, str):
        args.gpus = [int(gpu) for gpu in args.gpus.split(",") if gpu]

    print('=====> Preparing data...')
    # Multi-cuda
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        batch_size = args.batch_size

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
    train_transform_org = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    transform_train = TwoCropTransformAdv(transform_train, train_transform_org)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    val_dataset = datasets.CIFAR10(root='data',
                                   train=False,
                                   transform=val_transform,
                                   download=True)

    val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=256, shuffle=False,
                num_workers=8, pin_memory=True)

    label_pseudo_train_list = []
    num_classes_list = [2, 10, 50, 100, 500]

    dict_name = 'data/{}_pseudo_labels.pkl'.format(args.cname)
    f = open(dict_name, 'rb')  # Pickle file is newly created where foo1.py is
    feat_label_dict = pickle.load(f)  # dump data to f
    f.close()
    for i in range(5):
        class_num = num_classes_list[i]
        key_train = 'pseudo_train_{}'.format(class_num)
        label_pseudo_train = feat_label_dict[key_train]
        label_pseudo_train_list.append(label_pseudo_train)

    train_dataset = CIFAR10IndexPseudoLabelEnsemble(root='data',
                                                    transform=transform_train,
                                                    pseudoLabel_002=label_pseudo_train_list[0],
                                                    pseudoLabel_010=label_pseudo_train_list[1],
                                                    pseudoLabel_050=label_pseudo_train_list[2],
                                                    pseudoLabel_100=label_pseudo_train_list[3],
                                                    pseudoLabel_500=label_pseudo_train_list[4],
                                                    download=True)
    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=n_gpu*4)
    if args.wandb:
        wandb_logger = WandbLogger(
                name="advssl-eval_pgd_bn-1_step_train_att-idea1",
                project="officialcode_lightning",
                entity="kaistssl",
            )
        wandb_logger.log_hyperparams(args)


    autoencoder = LitAutoEncoder(**args.__dict__)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger if args.wandb else None,
        plugins=DDPPlugin(find_unused_parameters=True) if args.accelerator == "ddp" else None,
    )

    # trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
    trainer.fit(autoencoder, train_loader, val_loader)


def simclr_anchor_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
    extra_pos_mask = None,
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        temperature (float): temperature factor for the loss. Defaults to 0.1.
        extra_pos_mask (Optional[torch.Tensor]): boolean mask containing extra positives other
            than normal across-view positives. Defaults to None.
    Returns:
        torch.Tensor: SimCLR loss.
    """

    # pos = torch.einsum("nc,nc->n", [z1, z2.detach()]).unsqueeze(-1)
    # neg = torch.einsum("nc,ck->nk", [z1, z2.detach()])
    # logits = torch.cat([pos, neg], dim=1)
    # logits /= temperature
    # # targets = torch.zeros(query.size(0), device=query.device, dtype=torch.long)

    # targets = torch.range(0, z1.size(0)-1, device=z1.device, dtype=torch.long)
    # loss = F.cross_entropy(neg, targets)


    # return F.cross_entropy(logits, targets)



    device = z1.device

    b = z1.size(0)
    z = torch.cat((z1, z2), dim=0)
    z = F.normalize(z, dim=-1)
    z1 = F.normalize(z1, dim=-1)

    # logits = torch.einsum("if, jf -> ij", z, z) / temperature
    logits = torch.einsum("if, jf -> ij", z, z.detach()) / temperature

    # logits = torch.einsum("if, jf -> ij", z1, z) / temperature
    # import ipdb; ipdb.set_trace()


    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
    pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
    pos_mask[:, b:].fill_diagonal_(True)
    # pos_mask[b:, :].fill_diagonal_(True)

    # if we have extra "positives"
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask, extra_pos_mask)

    # all matches excluding the main diagonal
    logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

    exp_logits = torch.exp(logits) * logit_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1)
    # mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    # loss
    loss = -mean_log_prob_pos.mean()
    return loss

# ================================================================== #
#                     Model, Loss and Optimizer                      #
# ================================================================== #

# PGD attack model
class AttackPGD(nn.Module):
    def __init__(self, model, config, ce_weight, radius):
        super(AttackPGD, self).__init__()
        self.model = model
        self.ce_weight = ce_weight
        self.radius = radius
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Plz use xent for loss function.'

    def forward(self, images_t1, images_t2, images_org, targets, criterion):
        x1 = images_t1.clone().detach()
        x2 = images_t2.clone().detach()
        x_cl = images_org.clone().detach()
        # x_ce = images_org.clone().detach()
        x_ce = None

        images_org_high = generate_high(x_cl.clone(), r=self.radius)
        # x_HFC = images_org_high.clone().detach()
        x_HFC = None

        if self.rand:
            x_cl = x_cl + torch.zeros_like(x1).uniform_(-self.epsilon, self.epsilon)
            # x_ce = x_ce + torch.zeros_like(x1).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            x_cl.requires_grad_()
            # x_ce.requires_grad_()
            with torch.enable_grad():
                f_proj, f_pred = self.model(x_cl, bn_name='pgd', contrast=True)
                # fce_proj, fce_pred, logits_ce = self.model(x_ce, bn_name='pgd_ce', contrast=True, CF=True, return_logits=True, nonlinear=False)
                f1_proj, f1_pred = self.model(x1, bn_name='normal', contrast=True)
                f2_proj, f2_pred = self.model(x2, bn_name='normal', contrast=True)
                # f_high_proj, f_high_pred = self.model(x_HFC, bn_name='normal', contrast=True)
                # features = torch.cat([f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_high_proj.unsqueeze(1)], dim=1)
                # features = torch.cat([f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1)], dim=1)
                # loss_contrast = criterion(features)
                loss_contrast = simclr_anchor_loss_func(f_proj, f1_proj) 
                loss_ce = 0
                # for label_idx in range(5):
                #     tgt = targets[label_idx].long()
                #     lgt = logits_ce[label_idx]
                #     loss_ce += F.cross_entropy(lgt, tgt, size_average=False, ignore_index=-1) / 5.
                loss = loss_contrast + loss_ce * self.ce_weight
            # grad_x_cl, grad_x_ce = torch.autograd.grad(loss, [x_cl, x_ce])
            grad_x_cl = torch.autograd.grad(loss, [x_cl])[0]


            x_cl = x_cl.detach() + self.step_size * torch.sign(grad_x_cl.detach())
            x_cl = torch.min(torch.max(x_cl, images_org - self.epsilon), images_org + self.epsilon)
            x_cl = torch.clamp(x_cl, 0, 1)
            # x_ce = x_ce.detach() + self.step_size * torch.sign(grad_x_ce.detach())
            # x_ce = torch.min(torch.max(x_ce, images_org - self.epsilon), images_org + self.epsilon)
            # x_ce = torch.clamp(x_ce, 0, 1)
        return x1, x2, x_cl, x_ce, x_HFC


if __name__ == "__main__":
    main()
