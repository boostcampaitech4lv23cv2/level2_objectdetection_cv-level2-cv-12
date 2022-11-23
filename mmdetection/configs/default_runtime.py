project_name = 'inseo_test'
experiment_name = 'focal loss' 
work_dir = './work_dirs/'+project_name+'/'+experiment_name # save path 설정
#work_dir = './work_dirs/eok'

checkpoint_config = dict(interval=1) # 1에폭에 한번 저장
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
            init_kwargs={'project': project_name,
                        'name' : experiment_name,
                        'entity' : 'cv12'},
            interval=100, # Logging interval 
            log_checkpoint=False, # Save the checkpoint at every checkpoint interval as W&B Artifacts
            log_checkpoint_metadata=True,  # Log the evaluation metrics computed on the validation data with the checkpoint
            num_eval_images=100, # The number of validation images to be logged.
            bbox_score_thr=0.3
            ),
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)