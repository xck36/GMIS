# GMIS
The source code in this repository could generate the instance segmentation results based on ECCV 2018 paper: Affinity Derivation and Graph Merge for Instance Segmentation.https://arxiv.org/abs/1811.10870

Dependicies:
1. tensorflow
2. pillow
3. scikit-image

Example command line to run this program is
python ./demo.py --sem_ckpt_path=../model/semantic.ckpt --aff_ckpt_path=../model/affinity.ckpt --demo_data_dir=../cityscapes_val_png --out_dir=../out --tmp_dir=/media/ramdisk --evaluation_result_out_dir=../eval_out --post_processing_exec=./post_processing/a.out --semantic_stride=8 --affinity_stride=8 --inference_process_num=4 --post_processing_process_num=16 --fresh_run --show_mask_on_image --use_sem_flip --use_aff_flip --auto_upsample --bike_prob_refine
After getting results, you may evaluate the results by cityscapes evaluation script.

Please be noted that:
1. The tensorflow may output slightly different results for different run. Thus, the final mAP result may also be a little different.
2. The models could be downloaded from: https://1drv.ms/f/s!ArfeUchTqlW6hQQBUVwqLcgTBkoH
3. The models provided above have been finetuned with output stride equal to 8. Thus, setting output stride to 16 cannot get the same results in the paper.
