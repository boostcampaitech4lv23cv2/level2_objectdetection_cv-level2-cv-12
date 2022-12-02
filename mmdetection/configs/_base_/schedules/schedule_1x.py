# optimizer
#learning_rate = 1e-5

optimizer = dict(type='Adam', lr=0.00005, weight_decay=0.00001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    #gamma=0.2,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 15])
runner = dict(type='EpochBasedRunner', max_epochs=20)


# config cascade JJ
"""
learning_rate = 1e-4
optimizer = dict(type='AdamW', lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7, 14, 21, 28])
runner = dict(type='EpochBasedRunner', max_epochs=36)
"""
# CosineAnnealing schedule
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1.0 / 10,
#     min_lr_ratio=1e-5)