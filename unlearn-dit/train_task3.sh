source /home/jichang/workspace/.dm_env/bin/activate

export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1

########### Task 3 ############
CUDA_VISIBLE_DEVICES=3 torchrun --nnodes=1 --master_port=29502 \
        --nproc_per_node=1 train_cl.py \
        --num-workers 16 \
        --model DiT-B/2 \
        --data-path dataset/canvas10 \
        --num-classes 30 \
        --epochs 50000 \
        --global-batch-size 16 \
        --lr 0.0002 \
        --log-every 10 \
        --ckpt-every 5000 \
        --results-dir model/task3 \
        --image-size 128 \
        --subfolders Cats Dogs Rabbits \
        --cl \
        --ckpt-path /model/task1/../0100000.pt  \
        --ckpt-lora-path /model/task2/../0100000.pt  \
        --old-num-classes-begin 6 \
        --old-num-classes 6 \
        --cl-num-classes-begin 12 \
        --cl-num-classes 6 \
        --k-gr 3 \
        --lora \
        --lora-r 4 \
        --multi-lora \
        --convnext \
        --test
