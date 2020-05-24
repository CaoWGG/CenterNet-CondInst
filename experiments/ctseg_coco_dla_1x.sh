cd src
# train
python main.py ctseg --exp_id coco_dla_1x --batch_size 20 --master_batch 9 --lr 1.25e-4 --gpus 0,1 --num_workers 4
# test
python test.py ctseg --exp_id coco_dla_1x --keep_res --resume
cd ..
