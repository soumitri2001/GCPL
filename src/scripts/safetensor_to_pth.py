import os, sys
import torch
from safetensors.torch import load_file

'''utility script to convert ./.cache/directory/*.safetensors ==> ./saved_embeds/directory/*.pth '''

gamma = [0.0]
dsets = ['lungcolon', 'colorectal', 'skin', 'fractal']

# src_dir, dest_dir = sys.argv[1], sys.argv[2]

src, dest = ".cache_mucti_v1/{}/g{}_f32_bs1/embeds", "saved_embeds/mucti_v1/g{}_f32_bs1/{}-train-16shot/"

# if not os.path.exists(dest):
#     os.makedirs(dest)

for g in gamma:
    for ds in dsets:
        src_dir, dest_dir = src.format(ds, g), dest.format(g, ds)
        # os.makedirs(dest_dir, exist_ok=True)
        for token_emb in os.listdir(src_dir):
            if 'ckpt' in token_emb:
                token_name, ckpt_itr = token_emb.split('-learned_embeds')[0], token_emb.split('ckpt')[-1].split('.safetensors')[0]
                # if ckpt_itr not in ['1000', '2000', '3000', '4000']:

                dest_dir2 = dest_dir + "/" + f"ep_{ckpt_itr}"
                os.makedirs(dest_dir2, exist_ok=True)
            else:
                dest_dir2 = dest_dir 
                assert 'learned_embeds.safetensors' in token_emb, AssertionError("ERROR! Check directory paths again")
                token_name = token_emb.split('-learned_embeds')[0]
            assert token_name[0] == '<' and token_name[-1] == '>', AssertionError("ERROR! Token name incorrect")
            emb_file = load_file(os.path.join(src_dir, token_emb))
            emb_tensor = emb_file[token_name]
            print(f'token of embedding: {emb_file.keys()} | shape of embedding: {emb_tensor.shape}')
            torch.save(emb_tensor, os.path.join(dest_dir2, token_name + ".pth"))