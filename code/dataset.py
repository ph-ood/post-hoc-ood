import utils
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):

    def __init__(self, path_base, df, img_transform, postprocess = False, labelled = True, return_path = False):
        self.path_base = path_base
        self.df = df
        self.img_transform = img_transform
        self.labelled = labelled
        self.postprocess = postprocess
        self.return_path = return_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        loc = row["path"]
    
        img = utils.load(f"{self.path_base}/{loc}", gray = False) # [h, w, c]

        if self.labelled:
            label = int(row["label"])

        img = Image.fromarray(img) # convert to PIL
        if self.img_transform:
            img = self.img_transform(img)

        if self.postprocess:
            img = utils.zScore(img)

        if self.return_path:
            if self.labelled:
                return img, label, loc
            else:
                return img, loc
        else:
            if self.labelled:
                return img, label
            else:
                return img