import argparse
import os
import glob
import pandas as pd

'''this script converts classname-based prompts into TI token-based prompts'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='dataset')
    args = parser.parse_args()

    DATASET = args.d

    # DATASET = 'skin' # replace this for each dataset

    src_csv_file = "./prompts/{}_prompts.csv"
    dest_csv_file = "./prompts_textinv/{}_prompts.csv"

    prompts_df = pd.read_csv(src_csv_file.format(DATASET))
    new_df = {
        'prompt' : [],
        'classname' : [],
        'classidx' : []
    }

    for i in range(len(prompts_df)):
        prompt, clsname, clsidx = prompts_df.iloc[i]['prompt'], prompts_df.iloc[i]['classname'], prompts_df.iloc[i]['classidx']
        token_clsname = "<" + clsname.replace(' ', '&') + ">"
        if DATASET in ['food', 'fractal', 'birds']: # different from other classname formats
            clsname = clsname.replace('_', ' ')
        new_prompt = prompt.replace(clsname, token_clsname)
        print(prompt, '==>', new_prompt)
        new_df['prompt'].append(new_prompt)
        new_df['classname'].append(token_clsname)
        new_df['classidx'].append(clsidx)

    pd.DataFrame(new_df).to_csv(dest_csv_file.format(DATASET), index=False)