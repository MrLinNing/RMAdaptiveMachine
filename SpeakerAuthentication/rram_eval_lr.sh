python -u rram_eval_basic_learn_full.py --mlp '140,512,5' \
    --batch_size 128 \
    --Tw 10 \
    --num_selected_speakers 5\
    --n_bins 5 \
    --name 'snn_lsm_512_5cls_10tw_best_model' \
    --pretrained_model './checkpoints/snn_lsm_512_5cls_10tw_best_model.pth' \