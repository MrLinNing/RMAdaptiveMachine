python -u rram_eval_basic_continual_full.py --mlp '140,512,6' \
    --batch_size 128 \
    --Tw 10 \
    --num_selected_speakers 7\
    --num_selected_speakers_before 5\
    --n_bins 5 \
    --previous_unlearn_idx 1 \
    --continue_learn_idx 1 \
    --name 'snn_lsm_512_full_continue_5cls_10twat38' \
    --pretrained_model './checkpoints_continue_full/snn_lsm_512_full_continue_5cls_10twat38.pth' \