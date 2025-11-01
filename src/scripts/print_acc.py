import argparse
import os.path as osp
import glob
import torch
from tqdm import tqdm


def mean_per_class_acc(correct, labels):
    total_acc = 0
    for cls in torch.unique(labels):
        mask = labels == cls
        total_acc += correct[mask].sum() / mask.sum()
    return total_acc / len(torch.unique(labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)
    args = parser.parse_args()

    # get list of files
    files = sorted(glob.glob(osp.join(args.folder, '**/*.pt'), recursive=True))

    preds = []
    labels = []
    for f in tqdm(files):
        data = torch.load(f)
        preds.append(data['pred'])
        labels.append(data['label'])
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    # top 1
    correct = (preds == labels).sum().item()
    print(f'Top 1 acc: {correct / len(preds) * 100:.2f}%')
    # mean per class
    print(f'Mean per class acc: {mean_per_class_acc(preds == labels, labels) * 100:.2f}%')


if __name__ == '__main__':
    main()
