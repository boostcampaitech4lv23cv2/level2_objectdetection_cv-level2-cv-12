# optimizer
optimizer_config = dict(grad_clip=None)

optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

# learning policy
# lr_config = dict(
#     policy='CosineRestart',
#     warmup='linear',
#     warmup_iters=1099,
#     warmup_ratio=0.001,
#     periods=[3912, 3912, 4890, 4890, 5868],
#     restart_weights=[1, 0.85, 0.75, 0.7, 0.6],
#     by_epoch=False,
#     min_lr=1e-5
#     )

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=978,
    warmup_ratio=0.001,
    step=[5, 9, 14, 19])

runner = dict(type='EpochBasedRunner', max_epochs=24)