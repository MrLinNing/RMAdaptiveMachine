source /home/jichang/workspace/.dm_env/bin/activate

export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1

########### Task 1 ############ 
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --master_port=29507 \
        --nproc_per_node=1 train_cl.py \
        --num-workers 16 \
        --model DiT-B/2 \
        --data-path dataset/canvas10 \
        --num-classes 30 \
        --epochs 20000 \
        --global-batch-size 16 \
        --lr 0.0002 \
        --log-every 10 \
        --ckpt-every 2000 \
        --results-dir model/task1 \
        --image-size 128 \
        --subfolders Cats Dogs Rabbits \
        --cl-num-classes-begin 0 \
        --cl-num-classes 12 \
        --lora \
        --lora-r 4 \
        --convnext \
        # --ckpt-path model/task1/0100000.pt  \
        # --test