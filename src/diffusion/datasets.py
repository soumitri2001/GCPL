import os.path as osp
from torchvision import datasets
from diffusion.utils import DATASET_ROOT, get_classes_templates
from diffusion.dataset.objectnet import ObjectNetBase
from diffusion.dataset.imagenet import ImageNet as ImageNetBase

from diffusion.dataset.cub import CUB200 
from diffusion.dataset.fractal import Fractal60
from diffusion.dataset.wikichurches import WikiChurches
from diffusion.dataset.yoga import Yoga82
from diffusion.dataset.colorectal import CRC5K
from diffusion.dataset.lung_colon import LC25K
from diffusion.dataset.isic_skinlesion import ISIC
from diffusion.dataset.insect_pest import IP102
from diffusion.dataset.cornseeds import CornSeed

class MNIST(datasets.MNIST):
    """Simple subclass to override the property"""
    class_to_idx = {str(i): i for i in range(10)}


def get_target_dataset(name: str, train=False, transform=None, target_transform=None):
    """Get the torchvision dataset that we want to use.
    If the dataset doesn't have a class_to_idx attribute, we add it.
    Also add a file-to-class map for evaluation
    """

    # new additions 
    if name == 'cars': # stanford-cars
        dataset = datasets.StanfordCars(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                        target_transform=target_transform, download=False)
        dataset._image_files = [sample[0] for sample in dataset._samples]
        dataset._labels = [sample[1] for sample in dataset._samples]
        dataset.file_to_class = {f.split("/")[-1].split(".")[0]: l for f, l in zip(dataset._image_files, dataset._labels)}

    elif name == 'dogs': # stanford-dogs
        raise ValueError(f"Dataset {name} not supported.")

    elif name == 'textures': # DTD
        dataset = datasets.DTD(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                               target_transform=target_transform, download=True)
        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._image_files, dataset._labels)}

    elif name == "pets":
        dataset = datasets.OxfordIIITPet(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                         target_transform=target_transform, download=True)
        dataset._image_files = dataset._images
        # lower case every key in the class_to_idx
        dataset.class_to_idx = {k.lower(): v for k, v in dataset.class_to_idx.items()}
        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._images, dataset._labels)}


    #### NEW DATASETS ####
    elif name == 'birds': # CUB
        dataset = CUB200(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                            target_transform=target_transform, download=False)
        dataset.file_to_class = {
            f.name.split(".")[0]: dataset.class_to_idx[f.parents[0].name]
            for f in dataset._image_files
        }
    
    elif name == 'fractal': # fractal-60
        dataset = Fractal60(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                            target_transform=target_transform, download=False)
        dataset.file_to_class = {
            f.name.split(".")[0]: dataset.class_to_idx[f.parents[0].name]
            for f in dataset._image_files
        }

    elif 'wikichurches' in name: # wikichurches_{version}
        version = name.split('_')[-1]
        assert version in ['wc4', 'wc6', 'wc14', 'wcH'], ValueError("Invalid version of Wikichurches provided!")
        dataset = WikiChurches(root=DATASET_ROOT, version=version, split="train" if train else "test", transform=transform,
                            target_transform=target_transform, download=False)
        dataset.file_to_class = {
            f.name.split('.')[0]: l for f, l in zip(dataset._image_files, dataset._labels)
        }

    elif name == 'posture': # yoga
        raise ValueError(f"Dataset {name} not supported.")

    elif name == 'colorectal': # colorectal
        dataset = CRC5K(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                            target_transform=target_transform, download=False)
        dataset.file_to_class = {
            f.name.split(".")[0]: dataset.class_to_idx[f.parents[0].name]
            for f in dataset._image_files
        }

    elif name == 'lungcolon': # LC25k
        dataset = LC25K(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                            target_transform=target_transform, download=False)
        dataset.file_to_class = {
            f.name.split(".")[0]: dataset.class_to_idx[f.parents[0].name]
            for f in dataset._image_files
        }

    elif name == 'skin': # ISIC2018
        dataset = ISIC(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                            target_transform=target_transform, download=False)
        dataset.file_to_class = {
            f.name.split(".")[0]: dataset.class_to_idx[f.parents[0].name]
            for f in dataset._image_files
        }
    
    elif name == 'insects':
        dataset = IP102(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                            target_transform=target_transform, download=False)
        dataset.file_to_class = {
            f.name.split(".")[0]: dataset.class_to_idx[f.parents[0].name]
            for f in dataset._image_files
        }

    elif name == 'seeds':
        dataset = CornSeed(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                            target_transform=target_transform, download=False)
        dataset.file_to_class = {
            f.name.split(".")[0]: l for f, l in zip(dataset._image_files, dataset._labels)
        }


    #### original code ####
    elif name == "cifar10":
        dataset = datasets.CIFAR10(root=DATASET_ROOT, train=train, transform=transform,
                                   target_transform=target_transform, download=True)
        dataset._images, dataset._labels = dataset.data, dataset.targets
    elif name == "stl10":
        dataset = datasets.STL10(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                 target_transform=target_transform, download=True)
        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.classes)}
    
    elif name == "flowers":
        dataset = datasets.Flowers102(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                      target_transform=target_transform, download=True)
        classes = list(get_classes_templates('flowers')[0].keys())  # in correct order
        dataset.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        dataset.file_to_class = {f.name.split('.')[0]: l for f, l in zip(dataset._image_files, dataset._labels)}
    
    elif name == "aircraft":
        dataset = datasets.FGVCAircraft(root=DATASET_ROOT, split="trainval" if train else "test", transform=transform,
                                        target_transform=target_transform, download=True)

        # replace backslash with underscore -> need to be dirs
        dataset.class_to_idx = {
            k.replace('/', '_'): v
            for k, v in dataset.class_to_idx.items()
        }

        dataset.file_to_class = {
            fn.split("/")[-1].split(".")[0]: lab
            for fn, lab in zip(dataset._image_files, dataset._labels)
        }
        # dataset.file_to_class = {
        #     fn.split("/")[-1].split(".")[0]: lab
        #     for fn, lab in zip(dataset._image_files, dataset._labels)
        # }

    elif name == "food":
        dataset = datasets.Food101(root=DATASET_ROOT, split="train" if train else "test", transform=transform,
                                   target_transform=target_transform, download=True)
        dataset.file_to_class = {
            f.name.split(".")[0]: dataset.class_to_idx[f.parents[0].name]
            for f in dataset._image_files
        }
    elif name == "eurosat":
        if train:
            raise ValueError("EuroSAT does not have a train split.")
        dataset = datasets.EuroSAT(root=DATASET_ROOT, transform=transform, target_transform=target_transform,
                                   download=True)
    elif name == 'imagenet':
        assert not train
        base = ImageNetBase(transform, location=DATASET_ROOT)
        dataset = datasets.ImageFolder(root=osp.join(DATASET_ROOT, 'imagenet/val'), transform=transform,
                                       target_transform=target_transform)
        dataset.class_to_idx = None  # {cls: i for i, cls in enumerate(base.classnames)}
        dataset.classes = base.classnames
        dataset.file_to_class = None
    elif name == 'objectnet':
        base = ObjectNetBase(transform, DATASET_ROOT)
        dataset = base.get_test_dataset()
        dataset.class_to_idx = dataset.label_map
        dataset.file_to_class = None  # todo
    elif name == "caltech101":
        if train:
            raise ValueError("Caltech101 does not have a train split.")
        dataset = datasets.Caltech101(root=DATASET_ROOT, target_type="category", transform=transform,
                                      target_transform=target_transform, download=True)

        dataset.class_to_idx = {cls: i for i, cls in enumerate(dataset.categories)}
        dataset.file_to_class = {str(idx): dataset.y[idx] for idx in range(len(dataset))}
    elif name == "mnist":
        dataset = MNIST(root=DATASET_ROOT, train=train, transform=transform, target_transform=target_transform,
                        download=True)
    else:
        raise ValueError(f"Dataset {name} not supported.")

    if name in {'mnist', 'cifar10', 'stl10', 'aircraft'}:
        dataset.file_to_class = {
            str(idx): dataset[idx][1]
            for idx in range(len(dataset))
        }

    assert hasattr(dataset, "class_to_idx"), f"Dataset {name} does not have a class_to_idx attribute."
    assert hasattr(dataset, "file_to_class"), f"Dataset {name} does not have a file_to_class attribute."
    return dataset
