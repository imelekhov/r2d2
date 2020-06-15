import numpy as np
import random
from PIL import Image
from datasets.dataset import Dataset
from datasets.pair_dataset import PairDataset
from os import path as osp


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
    def __init__(self, intra_style=True, txt_fn="data/train_phototourism_st_small_fltrd.txt"):
        Dataset.__init__(self)
        self.intra_style = intra_style
        self.path_orig_img = "/ssd/data/phototourism/orig"
        self.path_style_img = "/ssd/data/phototourism/style_transfer"
        self.txt_fn = txt_fn
        self.imgs = [line.rstrip("\n") for line in open(self.txt_fn)]
        self.nimg = len(self.imgs)
        self.npairs = len(self.imgs)

        self.orig_style_name = "orig"
        self.styles_transfer_dict = {"snow": "20170212_071659",
                                     "winter_twilight": "20170223_165410",
                                     "rainy": "20170501_110319",
                                     "night": "20141222_110812",
                                     }

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

    def get_pair(self, pair_idx, output=()):
        if isinstance(output, str): output = output.split()

        fname1 = self.get_key(pair_idx)
        fpath1 = self._get_img_base_path(fname1)
        img1 = self._read_image(osp.join(fpath1, fname1))
        if not self.intra_style:
            img2 = img1.copy()
        else: # not so trivial
            fname_chunks = [x.strip('_') for x in fname1.split("___")]
            styles = list(self.styles_transfer_dict.keys())
            if len(fname_chunks) == 1:
                style_2nd_img = random.choice(styles)
                fpath2 = fname1.split('/')[0] + '/' + fname1.split('/')[-1][:-4]
            else:
                first_style = fname_chunks[-1][:-4]
                styles.remove(first_style)
                styles += [self.orig_style_name]
                style_2nd_img = random.choice(styles)
                if style_2nd_img == self.orig_style_name:
                    orig_style_fname = fname_chunks[0].split('/')[-1] + '.jpg'
                    fpath2 = osp.join(fname1.split('/')[0], "dense", "images", orig_style_fname)
                else:
                    fpath2 = fname_chunks[0]

            if style_2nd_img in self.styles_transfer_dict:
                fname2 = fpath2 + "___" + self.styles_transfer_dict[style_2nd_img] + "___" + style_2nd_img + ".png"
            else:
                # the second image is the original image (without any style), i.e
                fname2 = fpath2
            fpath2 = self._get_img_base_path(fname2)
            img2 = self._read_image(osp.join(fpath2, fname2))

        W, H = img1.size
        sx = img2.size[0] / float(W)
        sy = img2.size[1] / float(H)

        meta = {}
        if 'aflow' in output or 'flow' in output:
            mgrid = np.mgrid[0:H, 0:W][::-1].transpose(1, 2, 0).astype(np.float32)
            meta['aflow'] = mgrid * (sx,sy)
            meta['flow'] = meta['aflow'] - mgrid

        if 'mask' in output:
            meta['mask'] = np.ones((H,W), np.uint8)

        if 'homography' in output:
            meta['homography'] = np.diag(np.float32([sx, sy, 1]))

        return img1, img2, meta
