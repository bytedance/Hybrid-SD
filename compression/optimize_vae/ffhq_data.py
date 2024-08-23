import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import crop
from PIL import Image
import random
import time

path = r"/mnt/bn/bytenn-yg2/datasets/ffhq1024/data"


class FFHQ(Dataset):
    def __init__(self, root=path):
        super().__init__()
        self.root = root 
        self.sub_dir = ["FFHQ-1024-1", "FFHQ-1024-2"]
        self.files = []
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for sub_dir in self.sub_dir:
            sub_path = os.path.join(self.root, sub_dir)
            for file in os.listdir(sub_path):
                if file.endswith(".png"):
                    self.files.append(os.path.join(sub_path, file))


    def __getitem__(self, index):

        retry = 3
        while retry > 0:
            try:
                image =  Image.open(self.files[index])
            except Exception as e:
                print("congestion, avoid errors caused by high latency in reading files")
                index = random.randint(0, self.__len__())
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        image = image.convert("RGB")
        image = transforms.Resize(512)(image)
        image = self.transform(image)
        return {"pixel_values":image}


    def __len__(self):
        return len(self.files)
    
def ffhq_dataloader(batch_size, num_worker=32, pretetch_factor=32):
    dataset = FFHQ()
    from prefetch_generator import BackgroundGenerator
    class DataLoaderX(DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())
    loader = DataLoaderX(
        dataset,
        num_workers=num_worker,
        batch_size=batch_size,
        prefetch_factor=pretetch_factor,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )
    return loader


if __name__ == "__main__":
    train_dataloader = ffhq_dataloader(batch_size=24)
    for i, batch in enumerate(train_dataloader):
        print(i, batch["pixel_values"].shape)
