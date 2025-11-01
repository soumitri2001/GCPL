import os
import shutil

dsets = sorted(os.listdir('.images_16shot_seed1'))
src, dest = '../diffusion-classifier-fewshot/.cache_mucti_v1/{}/images_16shot', '.images_16shot_seed1/{}'

for ds in dsets:
    src_path = src.format(ds)
    shutil.copytree(src_path, dest.format(ds), dirs_exist_ok=True)