from datetime import datetime
from pytz import timezone

project_name = 'test'
experiment_name = 'wandb setting' + datetime.now(timezone("Asia/Seoul")).strftime("(%m.%d %H:%M)")
work_dir = './work_dirs/'+project_name+'/'+experiment_name # save path 설정

checkpoint_config = dict(interval=1) # 1에폭에 한번 저장
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook', 
            interval=100, # logging 하는 step
            init_kwargs={ # wandb.init에 인자들
                'project': project_name, 
                'name' : experiment_name,
                'entity': 'cv12'})
        # dict(type='TensorboardLoggerHook')
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