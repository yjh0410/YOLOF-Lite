python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolof50 \
        -lr 0.06 \
        -lr_bk 0.02 \
        --batch_size 32 \
        --img_size 800 \
        --grad_clip_norm -1.0 \
        --num_workers 8 \
        --max_epoch 12 \
        --lr_epoch 8 11
