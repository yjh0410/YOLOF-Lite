python train.py \
        --cuda \
        -d voc \
        --root /mnt/share/ssd2/dataset/ \
        -v yolof53-DC5 \
        -lr 0.03 \
        -lr_bk 0.01 \
        --batch_size 16 \
        --img_size 640 \
        --grad_clip_norm 4.0 \
        --num_workers 8 \
        --schedule 3x \
        # --mosaic
