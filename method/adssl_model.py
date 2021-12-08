from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.resnet_cifar_multibn_ensembleFC import resnet18 as ResNet18

from losses import SupConLoss
from zk_utils.attack_model import AttackPGD
from zk_utils.acc_calculate import weighted_mean, accuracy_at_k
from losses import SupConLoss, OurLoss1, OurLoss2




def static_lr(
    get_lr, param_group_indexes, lrs_to_replace
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class ADSSL(pl.LightningModule):
    def __init__(self, 
    learning_rate, 
    decay, 
    nce_t, 
    ce_weight, 
    classifier_lr, 
    att_pgd_epsilon,
    att_pgd_step_size,
    att_pgd_num_steps,

    test_pgd_epsilon,
    test_pgd_num_steps,
    test_pgd_step_size,

    iter,
    min_lr,
    warmup_start_lr,
    warmup_epochs,
    radius,
    max_epochs,
    RA_test_interval,
    test_BN,
    train_loss_func,
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
        self.RA_test_interval = RA_test_interval
        self.test_BN = test_BN

        self.train_loss_func = train_loss_func

        # test pgd
        self.test_pgd_epsilon = test_pgd_epsilon
        self.test_pgd_num_steps = test_pgd_num_steps
        self.test_pgd_step_size = test_pgd_step_size


        config = {
            'epsilon': att_pgd_epsilon/255.,
            'num_steps': att_pgd_num_steps,
            'step_size': att_pgd_step_size/255.,
            'random_start': True,
            'loss_func': 'xent',
        }

        bn_names = ['normal', 'pgd', 'pgd_ce']

        self.model = ResNet18(bn_names=bn_names)
        self.net = AttackPGD(self.model, config, ce_weight, radius)

        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        if train_loss_func == 1:
            self.contrast_criterion = SupConLoss(temperature=nce_t)
        elif train_loss_func == 2:
            self.contrast_criterion = OurLoss1(temperature=nce_t)
        elif train_loss_func == 3:
            self.contrast_criterion = OurLoss2(temperature=nce_t)

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
        feats = self.model(X, bn_name=self.test_BN, return_feat=True)

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

        f_ori_proj, f2_pred = self.model(image_org, bn_name='normal', contrast=True)
        # # f_high_proj, f_high_pred = self.model(x_HFC, bn_name='normal', contrast=True)
        # features = torch.cat(
        #         [f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_high_proj.unsqueeze(1)], dim=1)
        features = torch.cat(
                [f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_ori_proj.unsqueeze(1)], dim=1)
        contrast_loss = self.contrast_criterion(features)
        

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

        
        metrics = {
            "batch_size": batch_size,
            "val_loss": out["loss"],
            "val_acc1": out["acc1"],
            "val_acc5": out["acc5"],}

        # robustness accuracy testing
        if self.current_epoch % self.RA_test_interval == 0:
            torch.set_grad_enabled(True)
            adv_images = self.PGD_val_adv(X, targets)
            torch.set_grad_enabled(False)
            out_ra = self._base_shared_step(adv_images, targets)

            metrics.update({"RA_val_loss": out_ra["loss"],
            "RA_val_acc1": out_ra["acc1"],
            "RA_val_acc5": out_ra["acc5"],
             })
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

        

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5,}

        if self.current_epoch % self.RA_test_interval == 0:
            ra_val_loss = weighted_mean(outs, "RA_val_loss", "batch_size")
            ra_val_acc1 = weighted_mean(outs, "RA_val_acc1", "batch_size")
            ra_val_acc5 = weighted_mean(outs, "RA_val_acc5", "batch_size")
            log.update({"RA_val_loss": ra_val_loss, "RA_val_acc1": ra_val_acc1, "RA_val_acc5": ra_val_acc5})

        self.log_dict(log, sync_dist=True)

    def PGD_val_adv(self, images, labels):
        r"""
        Overridden.
        """
        # images = images.clone().detach().to(self.device)
        # labels = labels.clone().detach().to(self.device)


        loss = nn.CrossEntropyLoss()

        steps = self.test_pgd_num_steps
        alpha = self.test_pgd_step_size / 255
        eps = self.test_pgd_epsilon / 255

        adv_images = images.clone().detach()

        # if self.random_start:
            # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()



        for _ in range(steps):
            adv_images.requires_grad = True

            # outputs = self.model(adv_images, bn_name='normal', return_feat=True)
            outputs = self.model(adv_images, bn_name=self.test_BN, return_feat=True)

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
