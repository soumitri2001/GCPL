import json
from pathlib import Path
from typing import Any, Callable, Tuple

from torchvision.datasets import Food101

class IP102(Food101):
    def __init__(
        self, 
        root: str, 
        split: str = "train", 
        transform: Callable[..., Any] = None, 
        target_transform: Callable[..., Any] = None, 
        download: bool = False
    ) -> None:
        super().__init__(root, split, transform=transform, target_transform=target_transform, download=False)
        self._base_folder = Path(self.root) / "IP102"
        self._meta_folder = self._base_folder / "meta"
        self._images_folder = self._base_folder / "images"

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")
        
        # test set too big, evaluate on val set instead (also big!!)
        if split == 'test': 
            split = 'val'

        self._labels = []
        self._image_files = []
        with open(self._meta_folder / f"{split}.json") as f:
            metadata = json.loads(f.read())
    
        self.classes = sorted(metadata.keys())
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        for class_label, im_rel_paths in metadata.items():
            self._labels += [self.class_to_idx[class_label]] * len(im_rel_paths)
            self._image_files += [
                self._base_folder.joinpath(im_rel_path) for im_rel_path in im_rel_paths
            ]

    def __len__(self) -> int:
        return super().__len__()
    
    def __getitem__(self, idx: int) -> Tuple[Any]:
        return super().__getitem__(idx)