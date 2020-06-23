import numpy as np
import random
from PIL import Image
from datasets.dataset import Dataset
from datasets.pair_dataset import PairDataset
from os import path as osp

from utils import get_style2fnames


class PhototourismImages(Dataset):
    def __init__(self, root="/ssd/data/phototourism/orig", txt_fn="data/train_phototourism_ms.txt"):
        Dataset.__init__(self)
        self.root = root
        self.txt_fn = txt_fn
        self.imgs = [line.rstrip("\n") for line in open(self.txt_fn)]
        self.nimg = len(self.imgs)

    def get_filename(self, img_idx, root=None):
        return osp.join(root or self.root, self.get_key(img_idx))

    def get_key(self, img_idx):
        return self.imgs[img_idx]


class PhototourismStyleImages(PairDataset):
    def __init__(self, txt_fn="data/train_phototourism_ms.txt", style2style=False):
        Dataset.__init__(self)
        self.txt_fn = txt_fn
        self.style2style = style2style
        self.path_orig_img = "/ssd/data/phototourism/orig"
        self.path_style_img = "/data/datasets/phototourism/style_transfer_all"
        self.style2fnames = get_style2fnames()

        # a dirty hack
        self.crop_size = 192

        self.imgs = [line.rstrip("\n") for line in open(self.txt_fn)]
        self.nimg = len(self.imgs)
        self.npairs = len(self.imgs)

    def _get_img_base_path(self, fname):
        return self.path_style_img if len(fname.split("___")) == 3 else self.path_orig_img

    def get_key(self, img_idx):
        return self.imgs[img_idx]

    def _read_image(self, fpath):
        try:
            return Image.open(fpath).convert('RGB')
        except Exception as e:
            raise IOError("Could not load image %s (reason: %s)" % (fpath, str(e)))
            sys.exit()

    def _get_pair_orig2style(self, idx):
        fpath1 = self.imgs[idx]
        img1 = self._read_image(osp.join(self.path_orig_img, fpath1))
        scene, _, _, fname1 = fpath1.split("/")
        # let us randomly choose one of the styles
        style_2nd_img = random.choice(list(self.style2fnames.keys()))
        if style_2nd_img == "orig":
            img2 = img1.copy()
        else:
            fname2 = fname1[:-4] + "___" + random.choice(self.style2fnames[style_2nd_img])[:-4] + ".png"
            img2 = self._read_image(osp.join(self.path_style_img, scene, style_2nd_img, fname2))
        return img1, img2

    def get_pair(self, pair_idx, output=()):
        if isinstance(output, str): output = output.split()

        if self.style2style:
            # let us randomly choose one of the styles
            style_1st_img = random.choice(list(self.style2fnames.keys()))
            if style_1st_img == "orig":
                img1, img2 = self._get_pair_orig2style(pair_idx)
            else:
                fpath1_orig = self.imgs[pair_idx]
                scene, _, _, fname1_orig = fpath1_orig.split("/")
                fname1 = fname1_orig[:-4] + "___" + random.choice(self.style2fnames[style_1st_img])[:-4] + ".png"
                img1 = self._read_image(osp.join(self.path_style_img, scene, style_1st_img, fname1))

                styles = list(self.style2fnames.keys())
                styles.remove(style_1st_img)
                style_2nd_img = random.choice(styles)
                if style_2nd_img == "orig":
                    img2 = self._read_image(osp.join(self.path_orig_img, fpath1_orig))
                else:
                    fname2 = fname1_orig[:-4] + "___" + random.choice(self.style2fnames[style_2nd_img])[:-4] + ".png"
                    img2 = self._read_image(osp.join(self.path_style_img, scene, style_2nd_img, fname2))

        else:
            img1, img2 = self._get_pair_orig2style(pair_idx)

        W, H = img1.size
        sx = img2.size[0] / float(W)
        sy = img2.size[1] / float(H)

        s = float(min(min(W, H), self.crop_size))
        if s != self.crop_size:
            # compute the ratio
            r = int(np.floor(self.crop_size / s) + 1)
            W, H = W * r, H * r
            img1 = img1.resize((W, H), Image.BILINEAR)
            img2 = img2.resize((W, H), Image.BILINEAR)

        meta = {}
        if 'aflow' in output or 'flow' in output:
            mgrid = np.mgrid[0:H, 0:W][::-1].transpose(1, 2, 0).astype(np.float32)
            meta['aflow'] = mgrid * (sx, sy)
            meta['flow'] = meta['aflow'] - mgrid

        if 'mask' in output:
            meta['mask'] = np.ones((H, W), np.uint8)

        if 'homography' in output:
            meta['homography'] = np.diag(np.float32([sx, sy, 1]))

        return img1, img2, meta
