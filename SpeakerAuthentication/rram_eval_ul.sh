python -u rram_eval_basic_unlearn_full.py --mlp '140,512,5' \
    --batch_size 128 \
    --Tw 10 \
    --num_selected_speakers 5\
    --n_bins 5 \
    --unlearned_class_idx 1 \
    --name 'snn_lsm_512_full_randlabel_5cls_10twat51' \
    --pretrained_model './checkpoints_randlabel_unlearn_full/snn_lsm_512_full_randlabel_5cls_10twat51.pth' \