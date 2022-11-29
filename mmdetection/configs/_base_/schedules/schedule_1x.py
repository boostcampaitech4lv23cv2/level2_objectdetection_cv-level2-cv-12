# optimizer
learning_rate = 1e-4
optimizer = dict(type='Adam', lr=learning_rate, weight_decay=learning_rate*0.1)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[5, 10, 15])
runner = dict(type='EpochBasedRunner', max_epochs=20)

# CosineAnnealing schedule
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1.0 / 10,
#     min_lr_ratio=1e-5)