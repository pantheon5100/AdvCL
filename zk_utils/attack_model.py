import torch
import torch.nn as nn
import torch.nn.functional as F

from fr_util import generate_high

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

        # images_org_high = generate_high(x_cl.clone(), r=self.radius)
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

                f_ori_proj, f2_pred = self.model(images_org, bn_name='normal', contrast=True)

                # f_high_proj, f_high_pred = self.model(x_HFC, bn_name='normal', contrast=True)
                # features = torch.cat([f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_high_proj.unsqueeze(1)], dim=1)
                
                # features = torch.cat([f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1)], dim=1)
                features = torch.cat([f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_ori_proj.unsqueeze(1)], dim=1)

                loss_contrast = criterion(features)


                # away from tau1 tau2
                # loss_contrast = -(F.cosine_similarity(f_proj, f1_proj, dim=-1).mean() + F.cosine_similarity(f_proj, f2_proj, dim=-1).mean())



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
