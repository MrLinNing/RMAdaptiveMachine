python -u train_randlabel_full.py --mlp '140,512,5' \
    --batch_size 128 \
    --epoch 60 \
    --lr 0.05 \
    --Tw 10 \
    --num_selected_speakers 5\
    --unlearned_class_idx 1\
    --n_bins 5 \
    --name 'snn_lsm_512_full_randlabel_5cls_10tw' \
    --pretrained_model './checkpoints/snn_lsm_512_5cls_10tw_best_model.pth'