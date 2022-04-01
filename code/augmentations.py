import utils
import torch
import numpy as np
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter, map_coordinates

class ElasticDistortion:

    def __init__(self, alpha = 150, sigma = 5):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, x):
        # x : [h, w, c]

        shape = x.shape
        h, w = shape[:2]
        grid = np.meshgrid(np.arange(h), np.arange(w), indexing = "ij")

        dh = self.alpha*gaussian_filter(np.random.uniform(-1, 1, size = (h, w)), self.sigma, mode = "constant", cval = 0)
        dw = self.alpha*gaussian_filter(np.random.uniform(-1, 1, size = (h, w)), self.sigma, mode = "constant", cval = 0)
        new_grid = np.reshape(grid[0] + dh, (-1, 1)), np.reshape(grid[1] + dw, (-1, 1))

        y = np.zeros(shape)
        for c in range(shape[-1]):
            y[..., c] = map_coordinates(x[..., c], new_grid, order = 1, mode = "nearest").reshape((h, w))   
        
        y = utils.float2uint(y)

        return y

class ShufflePatch:
  # Reference: https://stackoverflow.com/a/66963266

    def __init__(self, patch_size):
        self.p = patch_size

    def __call__(self, x):
        # x: [b, c, h, w]

        # Divide the batch of images into non-overlapping patches
        u = F.unfold(x, kernel_size = self.p, stride = self.p, padding = 0) # [b, p*p*c, total_patches]

        # Permute the patches of each image in the batch (each b is a patched image that's shuffled)
        plist = [b[:, torch.randperm(b.shape[-1])].unsqueeze(0) for b in u]
        pu = torch.cat(plist, dim = 0)

        # Fold the permuted patches back together
        f = F.fold(pu, x.shape[-2:], kernel_size = self.p, stride = self.p, padding = 0)

        return f

if __name__ == "__main__":

    import utils
    import numpy as np
    from config import *

    name = "distort"

    path = f"{PATH_DATA}/fmnist/train/0/1.png"
    img = utils.load(path)

    if name == "patch":
        asp = ShufflePatch(patch_size = 14)
        imgt = torch.tensor(img, dtype = torch.float32).permute(2, 0, 1).unsqueeze(0) # [1, 3, 28, 28]
        out = asp(imgt).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    elif name == "distort":
        ads = ElasticDistortion()
        out = ads(img)

    utils.save(out, f"{PATH_EXAMPLES}/{name}ed.png")