#!/bin/bash  

python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --seg --patch --stitch  

CUDA_VISIBLE_DEVICES=0 python extract_features_fp.py --data_h5_dir RESULTS_DIRECTORY --data_slide_dir DATA_DIRECTORY --csv_path RESULTS_DIRECTORY/process_list_autogen.csv --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs