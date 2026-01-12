source /home/jichang/workspace/.dm_env/bin/activate

export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
        
########### Task 2 ############
CUDA_VISIBLE_DEVICES=3 torchrun --nnodes=1 --master_port=29503 \
        --nproc_per_node=1 train_cl.py \
        --num-workers 64 \
        --model DiT-B/2 \
        --data-path dataset/canvas10 \
        --num-classes 30 \
        --epochs 20000 \
        --global-batch-size 16 \
        --lr 0.0002 \
        --log-every 10 \
        --ckpt-every 5000 \
        --results-dir model/task2 \
        --image-size 128 \
        --subfolders Cats Dogs Rabbits \
        --ul \
        --ckpt-path /model/task1/0100000.pt  \
        --data-path-gen dataset/canvas10  \
        --ul-num-classes-begin-gen 0 \
        --ul-num-classes-gen 12 \
        --ul-classes 0 1 2 3 4 5 \
        --lora \
        --lora-r 4 \
        --convnext \
        # --test