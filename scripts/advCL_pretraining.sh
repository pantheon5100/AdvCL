# python pretraining_advCL_lightning_lambda0.py --wandb \
#         --max_epochs 100 \
#         --name adv_att_5step-testBN_pgd \
#         --test_BN pgd \
#         --att_pgd_epsilon 8 \
#         --att_pgd_num_steps 5 \
#         --att_pgd_step_size 2 \
#         --test_pgd_epsilon 8 \
#         --test_pgd_num_steps 20 \
#         --test_pgd_step_size 2 \
#         --RA_test_interval 1 \
#         --gpus 3


# python pretraining_advCL_lightning_lambda0.py --wandb \
#         --max_epochs 100 \
#         --name adv_att_1step-test_BN_pgd \
#         --test_BN pgd \
#         --att_pgd_epsilon 8 \
#         --att_pgd_num_steps 1 \
#         --att_pgd_step_size 8 \
#         --test_pgd_epsilon 8 \
#         --test_pgd_num_steps 20 \
#         --test_pgd_step_size 2 \
#         --RA_test_interval 1 \
#         --gpus 0


# python pretraining_advCL_lightning_lambda0.py --wandb \
#         --max_epochs 100 \
#         --name adv_att_1step-test_BN_normal \
#         --test_BN normal \
#         --att_pgd_epsilon 8 \
#         --att_pgd_num_steps 1 \
#         --att_pgd_step_size 8 \
#         --test_pgd_epsilon 8 \
#         --test_pgd_num_steps 20 \
#         --test_pgd_step_size 2 \
#         --RA_test_interval 1 \
#         --gpus 4

# python pretraining_advCL_lightning_lambda0.py --wandb \
#         --max_epochs 100 \
#         --name adv_att_5step-test_BN_normal \
#         --test_BN normal \
#         --att_pgd_epsilon 8 \
#         --att_pgd_num_steps 5 \
#         --att_pgd_step_size 2 \
#         --test_pgd_epsilon 8 \
#         --test_pgd_num_steps 20 \
#         --test_pgd_step_size 2 \
#         --RA_test_interval 1 \
#         --gpus 1


# python pretraining_advCL_lightning_lambda0.py --wandb \
#         --max_epochs 100 \
#         --name adv_att_1step-test_BN_pgd-Loss_ourloss1 \
#         --test_BN pgd \
#         --att_pgd_epsilon 8 \
#         --att_pgd_num_steps 1 \
#         --att_pgd_step_size 8 \
#         --test_pgd_epsilon 8 \
#         --test_pgd_num_steps 20 \
#         --test_pgd_step_size 2 \
#         --RA_test_interval 1 \
#         --gpus 2

# python pretraining_advCL_lightning_lambda0.py --wandb \
#         --max_epochs 100 \
#         --name adv_att_1step-test_BN_pgd-Loss_ourloss2 \
#         --test_BN pgd \
#         --att_pgd_epsilon 8 \
#         --att_pgd_num_steps 1 \
#         --att_pgd_step_size 8 \
#         --test_pgd_epsilon 8 \
#         --test_pgd_num_steps 20 \
#         --test_pgd_step_size 2 \
#         --RA_test_interval 1 \
#         --gpus 1


python pretraining_advCL_lightning_lambda0.py --wandb \
        --max_epochs 100 \
        --name adv_att_1step-test_BN_pgd-loss_func3 \
        --test_BN pgd \
        --att_pgd_epsilon 8 \
        --att_pgd_num_steps 1 \
        --att_pgd_step_size 8 \
        --test_pgd_epsilon 8 \
        --test_pgd_num_steps 20 \
        --test_pgd_step_size 2 \
        --RA_test_interval 1 \
        --train_loss_func 3 \
        --gpus 1
