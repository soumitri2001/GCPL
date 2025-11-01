import argparse
import os.path as osp
import pandas as pd
from diffusion.datasets import get_target_dataset

# ROOT_DIR = '/mnt/opr/soumitri/diffusion-classifier'  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='dataset')
    args = parser.parse_args()

    dataset = args.d
    target_dataset = get_target_dataset(dataset)

    template_file = pd.read_json('diffusion/templates.json')
    classes, templates = template_file[dataset]['classes'], template_file[dataset]['templates']

    prompt = [templates[0].format(cls.replace('_', ' ')) for cls in classes]
    classname = list(target_dataset.class_to_idx.keys())
    classidx = list(target_dataset.class_to_idx.values())

    # sanity checks
    assert len(prompt) == len(classname) == len(classidx)
    # for i in range(len(prompt)):
    #     assert classname[i].lower().replace('_', '/') in prompt[i].lower(), f"{classname[i]} not found in {prompt[i].lower()}"

    # make pandas dataframe
    df = pd.DataFrame(data=dict(prompt=prompt,
                                classname=classname,
                                classidx=classidx))
    # save to csv
    df.to_csv(f'prompts/{dataset}_prompts.csv', index=False)
