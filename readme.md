### Run TIF

+ All code is in `main.py`
    + eval_tif: evaluate tif
    + eval_mpc: evaluate contrastive loss
    + eval_mpc_stage_1: evaluate stage 1
+ stage 1: Discriminative Information Amplificat: `stage1_trainer.py`
+ stage 2: Unstable Information Suppression: `stage2_trainer.py`
+ active learning: update model by add uncertainty samples in the last environment
+ create dataset: run `create_dataset.py`. I recommend you to use dataset I processed, and the feature names are contained in the folder. (I didn't check whether the code follows the same seed.)

Because roast server has been reset, current results are trained recently without hyperparameter tuning. IRM really relies parameter selection. But at least it still outperforms baselines.
Dataset is saved under: 
+ processed dataset: /scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features
+ raw dataset with original drebin feature: /scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/drebin
