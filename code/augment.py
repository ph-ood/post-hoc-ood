import torch
import numpy as np
import torch.nn.functional as F

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
    from config import *

    asp = ShufflePatch(patch_size = 14)
    path = f"{PATH_DATA}/mnist/train/5/0.png"

    img = utils.load(path)
    imgt = torch.tensor(img, dtype = torch.float32).permute(2, 0, 1).unsqueeze(0) # [1, 3, 28, 28]

    shuf = asp(imgt).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    utils.save(shuf, f"{PATH_EXAMPLES}/patched.png")