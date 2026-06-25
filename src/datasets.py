import json
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset
from config import IMAGE_DIRS, SPLIT_FILES

class SplitDataset(Dataset):
    def __init__(self, dataset_name: str, split: str, transform=None):
        split_path = SPLIT_FILES[dataset_name]
        self.image_root = IMAGE_DIRS[dataset_name]
        self.transform = transform
        with open(split_path) as f:
            entries = json.load(f)[split]
        self.samples: List[Tuple[Path, int, str]] = []
        for rel_path, class_id, class_name in entries:
            self.samples.append((self.image_root / rel_path, class_id, class_name))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, class_id, class_name = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, class_id, class_name
def get_class_names(dataset_name: str) -> List[str]:
    split_path = SPLIT_FILES[dataset_name]
    with open(split_path) as f:
        data = json.load(f)
    id_to_name = {}
    for split in ("train", "val", "test"):
        for _, class_id, class_name in data[split]:
            id_to_name[class_id] = class_name
    return [id_to_name[i] for i in sorted(id_to_name.keys())]
