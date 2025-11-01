import argparse
import numpy as np
import os
from datetime import datetime
import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import get_formatstr
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode

import warnings
warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}

def _index_of_token(token, array: list):
    return array.index(token)

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


def eval_prob_adaptive(unet, latent, text_embeds, scheduler, args, latent_size=64, all_noise=None):
    scheduler_config = get_scheduler_config(args)
    T = scheduler_config['num_train_timesteps']
    max_n_samples = max(args.n_samples)

    if all_noise is None:
        all_noise = torch.randn((max_n_samples * args.n_trials, 4, latent_size, latent_size), device=latent.device)
    if args.dtype == 'float16':
        all_noise = all_noise.half()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()

    data = dict()
    t_evaluated = set()
    remaining_prmpt_idxs = list(range(len(text_embeds)))
    start = T // max_n_samples // 2
    t_to_eval = list(range(start, T, T // max_n_samples))[:max_n_samples]

    for n_samples, n_to_keep in zip(args.n_samples, args.to_keep):
        ts = []
        noise_idxs = []
        text_embed_idxs = []
        curr_t_to_eval = t_to_eval[len(t_to_eval) // n_samples // 2::len(t_to_eval) // n_samples][:n_samples]
        curr_t_to_eval = [t for t in curr_t_to_eval if t not in t_evaluated]
        for prompt_i in remaining_prmpt_idxs:
            for t_idx, t in enumerate(curr_t_to_eval, start=len(t_evaluated)):
                ts.extend([t] * args.n_trials)
                noise_idxs.extend(list(range(args.n_trials * t_idx, args.n_trials * (t_idx + 1))))
                text_embed_idxs.extend([prompt_i] * args.n_trials)
        t_evaluated.update(curr_t_to_eval)
        pred_errors = eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
                                 text_embeds, text_embed_idxs, args.batch_size, args.dtype, args.loss)
        # match up computed errors to the data
        for prompt_i in remaining_prmpt_idxs:
            mask = torch.tensor(text_embed_idxs) == prompt_i
            prompt_ts = torch.tensor(ts)[mask]
            prompt_pred_errors = pred_errors[mask]
            if prompt_i not in data:
                data[prompt_i] = dict(t=prompt_ts, pred_errors=prompt_pred_errors)
            else:
                data[prompt_i]['t'] = torch.cat([data[prompt_i]['t'], prompt_ts])
                data[prompt_i]['pred_errors'] = torch.cat([data[prompt_i]['pred_errors'], prompt_pred_errors])

        # compute the next remaining idxs
        errors = [-data[prompt_i]['pred_errors'].mean() for prompt_i in remaining_prmpt_idxs]
        best_idxs = torch.topk(torch.tensor(errors), k=n_to_keep, dim=0).indices.tolist()
        remaining_prmpt_idxs = [remaining_prmpt_idxs[i] for i in best_idxs]

    # organize the output
    assert len(remaining_prmpt_idxs) == 1
    pred_idx = remaining_prmpt_idxs[0]

    return pred_idx, data


def eval_error(unet, scheduler, latent, all_noise, ts, noise_idxs,
               text_embeds, text_embed_idxs, batch_size, dtype='float32', loss='l2'):
    assert len(ts) == len(noise_idxs) == len(text_embed_idxs)
    pred_errors = torch.zeros(len(ts), device='cpu')
    idx = 0
    with torch.inference_mode():
        for _ in tqdm.trange(len(ts) // batch_size + int(len(ts) % batch_size != 0), leave=False):
            batch_ts = torch.tensor(ts[idx: idx + batch_size])
            noise = all_noise[noise_idxs[idx: idx + batch_size]]
            noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                            noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
            t_input = batch_ts.to(device).half() if dtype == 'float16' else batch_ts.to(device)
            text_input = text_embeds[text_embed_idxs[idx: idx + batch_size]]
            noise_pred = unet(noised_latent, t_input, encoder_hidden_states=text_input).sample
            if loss == 'l2':
                error = F.mse_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'l1':
                error = F.l1_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            elif loss == 'huber':
                error = F.huber_loss(noise, noise_pred, reduction='none').mean(dim=(1, 2, 3))
            else:
                raise NotImplementedError
            pred_errors[idx: idx + len(batch_ts)] = error.detach().cpu()
            idx += len(batch_ts)
    return pred_errors


def main():
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument('--dataset', type=str, default='cars',
                        choices=['stl10', 'mnist', 'cifar10', 'caltech101', 'imagenet', 'objectnet', 
                        'dogs', 'food', 'pets', 'flowers', 'textures', 'aircraft', 'cars',
                        'birds', 'insects', 'fractal', 'churches', 'posture',
                        'colorectal', 'skin', 'lungcolon', 'seeds',
                        'wikichurches_wc4', 'wikichurches_wc6', 'wikichurches_wc14', 'wikichurches_wcH'], 
                        help='Dataset to use')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Name of split')

    # run args
    parser.add_argument('--version', type=str, default='1-4', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(128, 256, 512), help='Number of trials per timestep')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials per timestep')
    parser.add_argument('--embeds_path', type=str, required=True, help='path/to/saved/embeddings')
    parser.add_argument('--noise_path', type=str, default=None, help='Path to shared noise to use')
    parser.add_argument('--subset_path', type=str, default=None, help='Path to subset of images to evaluate')
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--load_stats', type=bool, default=True, help='Load saved stats to compute acc')
    parser.add_argument('--loss', type=str, default='l1', choices=('l1', 'l2', 'huber'), help='Type of loss to use')

    parser.add_argument('--extra', type=str, default=None, help='To append to the run folder name')
    parser.add_argument('--method', type=str, default='vanilla', help='which method to use -- vanilla, mucti_v1, mucti_v2 ...')
    parser.add_argument('--suffix', type=str, default=None, help='append a suffix in dir path; default adds nothing')
    parser.add_argument('--iters', type=int, default=0, help='which checkpoint embedding to use; default: no checkpoint')

    # multiple embed args
    parser.add_argument('--multi_vec', action='store_true', help='if the saved embedding is of multiple vectors')
    parser.add_argument('--mean_vec', action='store_true', help='if multiple vectors are to be averaged')

    # args for distributed inference
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use') # 0-based indexing
    
    # args for adaptively choosing which classes to continue trying
    parser.add_argument('--to_keep', nargs='+', type=int, required=True)
    parser.add_argument('--n_samples', nargs='+', type=int, required=True)

    args = parser.parse_args()
    assert len(args.to_keep) == len(args.n_samples)  

    # make run output folder
    OUT_DIR = f'fewshot_results/{args.method}'
    name = f"v{args.version}_{args.n_trials}trials_"
    name += '_'.join(map(str, args.to_keep)) + 'keep_'
    name += '_'.join(map(str, args.n_samples)) + 'samples'
    if args.interpolation != 'bicubic':
        name += f'_{args.interpolation}'
    name += f'_{args.loss}'
    if args.img_size != 512:
        name += f'_{args.img_size}'
    suffix = 'ep_final' if args.iters == 0 else f'ep_{args.iters}'

    if 'vanilla_rerun' in args.embeds_path:
        OUT_DIR = args.embeds_path.replace("saved_embeds", "fewshot_results")
        run_folder = osp.join(OUT_DIR, name, f"{args.worker_idx + 1}_{args.n_workers}")
    
    elif 'multiclass' in args.embeds_path:
        OUT_DIR = args.embeds_path.replace('saved_embeds', 'fewshot_results')
        run_folder = osp.join(OUT_DIR, name, f"{args.worker_idx + 1}_{args.n_workers}")
        
    else:
        if args.suffix is not None:
            suffix += f'_{args.suffix}'
        if args.extra is not None:
            run_folder = osp.join(OUT_DIR, args.dataset + '_' + args.extra, name, suffix)
        else:
            run_folder = osp.join(OUT_DIR, args.dataset, name, suffix)
    
    os.makedirs(run_folder, exist_ok=True)
    print(f'Run folder: {run_folder}')
    
    # set up dataset
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    latent_size = args.img_size // 8
    target_dataset = get_target_dataset(args.dataset, train=args.split == 'train', transform=transform)
    num_classes = len(target_dataset.class_to_idx)

    # load prompt file (### change to new csv file with TI token format)
    prompts_df = pd.read_csv(osp.join('prompts_textinv', f'{args.dataset}_prompts.csv'))
    assert 'prompt' in prompts_df.keys()
    
    # set up prompts
    template_file = pd.read_json('diffusion/templates.json')
    prompt_template = template_file[args.dataset]['templates'][0]

    # load pretrained SD model components
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    torch.backends.cudnn.benchmark = True

    # load noise
    if args.noise_path is not None:
        assert not args.zero_noise
        all_noise = torch.load(args.noise_path).to(device)
        print('Loaded noise from', args.noise_path)
    else:
        all_noise = None
    
    
    #### following code derived from: https://github.com/huggingface/diffusers/blob/v0.25.1/src/diffusers/loaders/textual_inversion.py#L401 ####
    '''this code follows the exact steps for loading TI embeds as in TI script'''

    is_multi_vector = False
    num_vectors = 1
    
    # 4. Retrieve tokens and embeddings
    assert args.method in args.embeds_path, AssertionError(f'{args.method} not found in entered embedding path: {args.embeds_path}')

    tokens, embeddings = [], []
    for token in prompts_df.classname.tolist():
        tokens.append(token)
        embedding = torch.load(osp.join(args.embeds_path, f'{token}.pth'))
        if len(embedding.shape) == 1:
            embedding = embedding.unsqueeze(0) 
        if embedding.shape[0] > 1:
            ### case 1 -- take mean of all embeds ==> single embed
            if args.multi_vec and args.mean_vec:
                embedding = torch.mean(embedding, dim=0).unsqueeze(0)
            ### case 2 -- use all embeds seperately
            else:
                is_multi_vector = True
                num_vectors = embedding.shape[0]
                tokens += [f"{token}_{i}" for i in range(1, num_vectors)]
        embeddings.append(embedding)
    embeddings = torch.cat(embeddings, dim=0) 

    assert len(tokens) == len(embeddings) == num_classes * num_vectors

    print("[DEBUG #1]", text_encoder.get_input_embeddings().weight.shape)

    # 6. Make sure all embeddings have the correct size
    expected_emb_dim = text_encoder.get_input_embeddings().weight.shape[-1]
    if any(expected_emb_dim != emb.shape[-1] for emb in embeddings):
        raise ValueError(
            f"Loaded embeddings are of incorrect shape. Expected each textual inversion embedding " \
            f"to be of shape {input_embeddings.shape[-1]}, but are {embeddings.shape[-1]} "
        )

    # 7.3 Increase token embedding matrix
    text_encoder.resize_token_embeddings(len(tokenizer) + len(tokens))
    input_embeddings = text_encoder.get_input_embeddings().weight

    # 7.4 Load token and embedding
    for token, embedding in zip(tokens, embeddings):
        # add tokens and get ids
        tokenizer.add_tokens(token)
        token_id = tokenizer.convert_tokens_to_ids(token)
        input_embeddings.data[token_id] = embedding

    input_embeddings.to(dtype=text_encoder.dtype, device=device)

    print("[DEBUG #2]", text_encoder.get_input_embeddings().weight.shape)

    # if multi-vector -- append <S*> with <S*>_1 <S*>_2 ... in prompts
    if is_multi_vector:
        for i in range(len(prompts_df)):
            orig_cls_token = tokens[i * num_vectors]
            # print("[DEBUG #3]", orig_cls_token, prompts_df.iloc[i]['prompt'])
            assert orig_cls_token in prompts_df.iloc[i]['prompt']
            new_cls_token = " ".join(tokens[i * num_vectors : (i+1) * num_vectors])
            prompts_df.prompt[i] = prompts_df.prompt[i].replace(orig_cls_token, new_cls_token)

    # print(prompts_df.prompt.tolist()[0:5])
    
    # obtain text embeddings for each prompt -- this follows original ZSDC code
    text_input = tokenizer(prompts_df.prompt.tolist(), padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")

    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(text_input.input_ids), 100):
            text_embeddings = text_encoder(
                text_input.input_ids[i: i + 100].to(device),
            )[0]
            embeddings.append(text_embeddings)

    text_embeddings = torch.cat(embeddings, dim=0)
    assert len(text_embeddings) == len(prompts_df)
    assert text_embeddings.shape == (num_classes, 77, 768)

    print('all prompt embeddings loaded!!')

    # add optimization lines -- https://github.com/diffusion-classifier/diffusion-classifier/issues/15
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # unet.enable_xformers_memory_efficient_attention()

    # exit(0)

    # if subset of dataset to evaluate
    if args.subset_path is not None:
        idxs = np.load(args.subset_path).tolist()
    else:
        idxs = list(range(len(target_dataset)))
    idxs_to_eval = idxs[args.worker_idx::args.n_workers]

    print(f"[DEBUG #3] Images to be evaluated on GPU index {args.worker_idx}: {len(idxs_to_eval)} / {len(target_dataset)}")

    # exit(0)

    # inference code
    formatstr = get_formatstr(len(target_dataset) - 1)
    correct = 0
    total = 0
    pbar = tqdm.tqdm(idxs_to_eval)
    for i in pbar:
        if total > 0:
            pbar.set_description(f'Acc: {100 * correct / total:.2f}%')
        fname = osp.join(run_folder, formatstr.format(i) + '.pt')
        if os.path.exists(fname):
            print('Skipping', i)
            if args.load_stats:
                data = torch.load(fname)
                correct += int(data['pred'] == data['label'])
                total += 1
            continue
        image, label = target_dataset[i]

        with torch.no_grad():
            img_input = image.to(device).unsqueeze(0)
            if args.dtype == 'float16':
                img_input = img_input.half()
            x0 = vae.encode(img_input).latent_dist.mean
            x0 *= 0.18215
        pred_idx, pred_errors = eval_prob_adaptive(unet, x0, text_embeddings, scheduler, args, latent_size, all_noise)
        pred = prompts_df.classidx[pred_idx]
        torch.save(dict(errors=pred_errors, pred=pred, label=label), fname)
        if pred == label:
            correct += 1
        total += 1


if __name__ == '__main__':
    main()
