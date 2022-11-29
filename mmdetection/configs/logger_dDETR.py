# work_dir 이랑 log_config 잘라내서 logger.py로 만들기
work_dir = './work_dirs/dDETR' # save path 설정

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
            init_kwargs={'project': 'sangho',
                         'name': 'dDETR',
                         'entity': 'cv12'},
            interval=100, # Logging interval
            log_checkpoint=False, # Save the checkpoint at every checkpoint interval as W&B Artifacts
            log_checkpoint_metadata=True,  # Log the evaluation metrics computed on the validation data with the checkpoint
            num_eval_images=20,
            bbox_score_thr=0.3), # The number of validation images to be logged.
    ])