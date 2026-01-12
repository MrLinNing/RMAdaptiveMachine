python -u train_ssc.py --mlp '140,512,5' \
    --batch_size 128 \
    --epoch 300 \
    --lr 0.05 \
    --Tw 10 \
    --num_selected_speakers 5\
    --n_bins 5 \
    --name 'snn_lsm_512_5cls_10tw'