python pretraining_advCL_lightning_lambda0.py --wandb \
        --max_epochs 100 \
        --name adv_att_1step-test_BN_pgd-add_ori \
        --test_BN pgd \
        --att_pgd_epsilon 8 \
        --att_pgd_num_steps 1 \
        --att_pgd_step_size 8 \
        --test_pgd_epsilon 8 \
        --test_pgd_num_steps 20 \
        --test_pgd_step_size 2 \
        --RA_test_interval 1 \
        --train_loss_func 1 \
        --gpus 2
