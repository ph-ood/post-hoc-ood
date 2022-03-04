import utils
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):

    def __init__(self, path_base, df, img_transform):
        self.path_base = path_base
        self.df = df
        self.img_transform = img_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        loc = row["path"]
    
        img = utils.load(f"{self.path_base}/{loc}", gray = False) # [h, w, c]
        label = row["label"]

        img = Image.fromarray(img) # convert to PIL
        if self.img_transform:
            img = self.img_transform(img)

        return img, label