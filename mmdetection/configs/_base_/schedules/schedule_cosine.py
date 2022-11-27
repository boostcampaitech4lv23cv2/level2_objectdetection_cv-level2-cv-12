# optimizer
learning_rate = 0.00005
optimizer = dict(type='Adam', lr=learning_rate, weight_decay=learning_rate*0.1)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    by_epoch=False,
    periods=[9770, 9770, 9770, 9770, 9770], # iteration 적기 → epoch * iter per epoch → 현재 5epoch 마다 반복되게 설정
    restart_weights=[1, 0.5, 0.1, 0.01, 0.001],  # cosine 주기마다 max lr 설정(periods랑 restart_weights 리스트 길이 같게 설정해야함)
    min_lr=1e-7) #5e-6

runner = dict(type='EpochBasedRunner', max_epochs=25)
