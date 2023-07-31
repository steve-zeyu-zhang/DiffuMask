# generating Synthetic data and saving Attention Map
# python3 ./Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py --classes bird --thread_num 8 --output ./DiffMask_VOC/ --image_number 15000

CUDA_VISIBLE_DEVICES=0 python3 ./Stable_Diffusion/parallel_generate_VOC_Attention_AnyClass.py --classes aeroplane --thread_num 4 --output ./DiffMask_VOC/ --image_number 8000 --MY_TOKEN '0mzLY4TtGUkaACnbL8geexjfmx7eteZYGVJQZVaqb33qTA4hzkSRZRbgF4qQ'
