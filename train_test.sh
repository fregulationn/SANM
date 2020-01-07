#!/usr/bin/env sh
GPU_ID=0
BATCH_SIZE=1
WORKER_NUMBER=0
LEARNING_RATE=0.0005

TRAIN=0
TEST=1

if  [ $TRAIN = 1 ]; then
	CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                    --dataset brainwash --net res50\
                    --bs 1 --nw 2\
                    --lr 0.0005 --lr_decay_step 6\
                    --cuda\
                    --dcr\
                    --mimic\
                    --epochs 8
                    # --use_tfb\
fi

if [ $TEST = 1 ]; then

    CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py \
                    --dataset brainwash --net res50\
                    --checksession 1 --checkepoch 7 --checkpoint 21907\
                    --attention\
                    --dcr\
                    --mimic\
                    --flags attention_mimic\
                    --cuda
fi



# CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py \
#                 --dataset brainwash --net res50\
#                 --checksession 1 --checkepoch 7 --checkpoint 21907\
#                 --cuda

# CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py \
#                 --dataset brainwash --net res50\
#                 --checksession 1 --checkepoch 7 --checkpoint 21907\
#                 --attention\
#                 --flags attention\
#                 --cuda

# CUDA_VISIBLE_DEVICES=$GPU_ID python test_net.py \
#                 --dataset brainwash --net res50\
#                 --checksession 1 --checkepoch 7 --checkpoint 21907\
#                 --dcr\
#                 --mimic\
#                 --flags mimic\
#                 --cuda
