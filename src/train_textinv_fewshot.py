import os
import os.path as osp
import shutil
import random
import argparse
import pandas as pd
import torch
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from diffusion.datasets import get_target_dataset
from safetensors.torch import load_file
from gcpl import run_gcpl_single_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PLACEHOLDER_TOKENS = {
    'food' : 'food',
    'pets' : 'pet',
    'textures' : 'texture',
    'aircraft' : 'aircraft',
    'birds' : 'bird',
    'cars' : 'car',
    'fractal' : 'fractal',
    'flowers' : 'flower',
    'dogs' : 'dog',
    'colorectal' : 'tissue',
    'skin' : 'skin',
    'lungcolon' : 'tissue',
    'yoga' : 'posture',
    'insects' : 'insect',
    'seeds' : 'seed',
    'wikichurches_wc4' :  'architecture',
    'wikichurches_wc6' :  'architecture',
    'wikichurches_wc14' :  'architecture',
    'wikichurches_wcH' :  'architecture',
}

def set_seed(seed):
    '''set fixed seed'''
    random.seed(seed)
    torch.manual_seed(seed)

def arrange_dataset_by_label(dataset, idx_to_class):
    dset = {}
    for i, label in enumerate(dataset._labels):
        if label in idx_to_class.keys(): 
            '''may not be present since it is subset of class (distributed)'''
            cls_name = idx_to_class[label]
            if not cls_name in dset.keys():
                dset[cls_name] = []
            dset[cls_name].append(i)
    return dset

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform

def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)


'''
args to modify manually:
--dataset
--n_shots
--resolution
--seed (for multiple runs)
--num_vectors
'''


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='cars',
                        choices=['stl10', 'mnist', 'cifar10', 'caltech101', 'imagenet', 'objectnet', 
                        'dogs', 'food', 'pets', 'flowers', 'textures', 'aircraft', 'cars',
                        'birds', 'insects', 'fractal', 'posture',
                        'colorectal', 'skin', 'lungcolon', 'seeds',
                        'wikichurches_wc4', 'wikichurches_wc6', 'wikichurches_wc14', 'wikichurches_wcH'], 
                        help='Dataset to use')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'], help='Name of split')

    # few-shot args
    parser.add_argument('--n_shots', type=int, default=16, choices=[1, 2, 4, 5, 8, 16], help='Nos of samples for support set')
    
    # other args
    parser.add_argument('--seed', type=int, default=1, help='set random seed value')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--resolution', type=int, default=512, choices=[128, 256, 512], help='image dimensions')

    # textual inversion args
    parser.add_argument('--num_vectors', type=int, default=1, help='nos of vectors to learn for TI')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size for TI')
    parser.add_argument('--lrate', type=float, default=5.0e-04, help='learning rate for TI')
    parser.add_argument('--iters', type=int, default=500, help='nos of training iterations for TI')

    # distributed training args -- divides into subsets of classes
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use') # 0-based indexing

    args = parser.parse_args()

    set_seed(args.seed)

    # setup dataset
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.resolution)
    dataset = get_target_dataset(args.dataset, train=args.split == 'train', transform=None) # no transforms here
    template_file = pd.read_json('diffusion/templates.json')
    templates = template_file[args.dataset]['templates'][0]

    idx_to_class = {v : k for k, v in dataset.class_to_idx.items()}

    # if distributed -- sample subset of classes
    idxs = list(range(len(dataset)))
    if args.n_workers > 1:
        idx_cls_subset = idxs[args.worker_idx::args.n_workers]
        idx_to_class = {k : v for k, v in idx_to_class.items() if k in idx_cls_subset}
    
    # few-shot sampling from dataset
    arranged_dataset = arrange_dataset_by_label(dataset, idx_to_class)

    fewshot_dataset = {}

    for cls_name in arranged_dataset.keys():
        indexes = arranged_dataset[cls_name]
        if len(indexes) >= args.n_shots:
            sampled_items = random.sample(indexes, args.n_shots)
        else:
            # if repeat:
            sampled_items = random.choices(indexes, k=args.n_shots)
            # else:
            #     sampled_items = items
        fewshot_dataset[cls_name] = sampled_items

    print('-' * 50)    
    print(f"==> Training on {len(fewshot_dataset)} out of {len(dataset.class_to_idx)} classes on GPU index {args.worker_idx + 1} / {args.n_workers}")
    print('-' * 50)    
    
    # create run output folder
    dest_root = osp.join('./saved_embeds/vanilla/rerun', f'num_vectors={args.num_vectors}', f'{args.dataset}-{args.split}-{args.n_shots}shot')
    os.makedirs(dest_root, exist_ok=True)
    
    # assuming fewshot_dataset contains support set indexes for each class that correspond to the original dataset 
    for cls_name in fewshot_dataset.keys():
        support_set_indexes = fewshot_dataset[cls_name]
        assert len(support_set_indexes) == args.n_shots

        # create cache folder to save support images
        root_dir = f'.cache_vanilla/images_{args.dataset}-{args.worker_idx}'
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        os.mkdir(root_dir)

        for idx in support_set_indexes:
            image_path = str(dataset._image_files[idx]) # path/to/files; change to _image_files for food/flowers/aircraft; _images for pets
            shutil.copy(image_path, root_dir)

        print(f'current support set class: {cls_name} | len of support set: {len(os.listdir(root_dir))}')

        # run textual inversion script using cache folder as the root directory
        log_dir = f'.cache_vanilla/debug/logs_{args.dataset}-{args.worker_idx}'
        placeholder_token = "<" + cls_name.replace(' ', '&') + ">"

        # just another hard-coded rectification
        if '/' in placeholder_token:
            placeholder_token = placeholder_token.replace('/', '-')

        print(f'class name: {cls_name} | token: {placeholder_token}')

        # check if this embed is already learned -- then skip
        if (placeholder_token + ".pth") in os.listdir(dest_root):
            print(f'{placeholder_token} embedding found in {dest_root}! Skipping...')
        
        else:
            '''params to be passed for running TI script'''
            gcpl_params = {
                # current args
                'dataset' : args.dataset,
                'n_shots' : args.n_shots,
                'n_workers' : args.n_workers,
                'worker_idx' : args.worker_idx,
                'seed' : args.seed,

                # textual inversion args
                'templates' : templates,
                'train_data_dir' : root_dir,
                'num_vectors' : args.num_vectors,
                'placeholder_token' : placeholder_token,
                'initializer_token' : PLACEHOLDER_TOKENS[args.dataset] if args.dataset in PLACEHOLDER_TOKENS.keys() else 'object',
                'output_dir' : log_dir,
                'train_batch_size' : args.batch_size,
                'learning_rate' : args.lrate,
                'max_train_steps' : args.iters,
                'lr_warmup_steps' : 0,
                'resolution' : args.resolution,
                'learnable_property' : "object", # default: 'object'; change to 'style' for texture
                'pretrained_model_name_or_path' : "CompVis/stable-diffusion-v1-4"
            }

            run_gcpl_single_class(new_args=gcpl_params)

            # move the saved embedding in a different folder and delete cache folder
            saved_embedding = load_file(osp.join(gcpl_params['output_dir'], 'learned_embeds.safetensors'))
            assert placeholder_token in saved_embedding.keys()
            embedding = saved_embedding[placeholder_token]
            print(f'token of embedding: {saved_embedding.keys()} | shape of embedding: {embedding.shape}')

            new_embed_path = osp.join(dest_root, f'{placeholder_token}.pth')
            torch.save(embedding, new_embed_path)
            print(f'embedding of {cls_name} saved at {new_embed_path}!!')
            
            try:
                shutil.rmtree(log_dir)
            except:
                # print("[ERROR]", os.listdir(log_dir))
                # shutil.rmtree(log_dir)
                print("[ERROR]", os.listdir(log_dir))

        shutil.rmtree(root_dir)


if __name__ == '__main__':
    main()