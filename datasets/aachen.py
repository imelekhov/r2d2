# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb
from os import path as osp
import random
import numpy as np
from PIL import Image

from .dataset import Dataset
from .pair_dataset import PairDataset, StillPairDataset


class AachenImages (Dataset):
    """ Loads all images from the Aachen Day-Night dataset 
    """
    def __init__(self, select='db day night', root='/ssd/data/r2d2_data/aachen'):
        Dataset.__init__(self)
        self.root = root
        self.img_dir = 'images_upright'
        self.select = set(select.split())
        assert self.select, 'Nothing was selected'
        
        self.imgs = []
        root = os.path.join(root, self.img_dir)
        for dirpath, _, filenames in os.walk(root):
            r = dirpath[len(root)+1:]
            if not(self.select & set(r.split('/'))): continue
            self.imgs += [os.path.join(r,f) for f in filenames if f.endswith('.jpg')]
        
        self.nimg = len(self.imgs)
        assert self.nimg, 'Empty Aachen dataset'

    def get_key(self, idx):
        return self.imgs[idx]



class AachenImages_DB (AachenImages):
    """ Only database (db) images.
    """
    def __init__(self, **kw):
        AachenImages.__init__(self, select='db', **kw)
        self.db_image_idxs = {self.get_tag(i) : i for i,f in enumerate(self.imgs)}
    
    def get_tag(self, idx): 
        # returns image tag == img number (name)
        return os.path.split( self.imgs[idx][:-4] )[1]



class AachenPairs_StyleTransferDayNight (AachenImages_DB, StillPairDataset):
    """ synthetic day-night pairs of images 
        (night images obtained using autoamtic style transfer from web night images)
    """
    def __init__(self, root='/ssd/data/r2d2_data/aachen/style_transfer_ours', **kw):
        StillPairDataset.__init__(self)
        AachenImages_DB.__init__(self, **kw)
        old_root = os.path.join(self.root, self.img_dir)
        self.root = os.path.commonprefix((old_root, root))
        self.img_dir = ''

        newpath = lambda folder, f: os.path.join(folder, f)[len(self.root):]
        self.imgs = [newpath(old_root, f) for f in self.imgs]

        self.image_pairs = []
        for fname in os.listdir(root):
            tag = fname.split('.jpg.st_')[0]
            self.image_pairs.append((self.db_image_idxs[tag], len(self.imgs)))
            self.imgs.append(newpath(root, fname))

        self.nimg = len(self.imgs)
        self.npairs = len(self.image_pairs)
        assert self.nimg and self.npairs


class AachenPairs_StyleTransferDayNight_Star(StillPairDataset):
    def __init__(self, root_img_path='/ssd/data/r2d2_data/aachen'):
        StillPairDataset.__init__(self)

        self.root_img_path = root_img_path
        self.path_orig_img = osp.join(self.root_img_path, "images_upright", "db")
        self.path_style_img = osp.join(self.root_img_path, "style_transfer_ours")

        self.styles_transfer_dict = {"snow": "20170212_071659",
                                     "winter_twilight": "20170223_165410",
                                     "rainy": "20170501_110319",
                                     "night": "20141222_110812",
                                     "orig": None,
                                     }

        self.imgs = os.listdir(self.path_style_img)
        self.nimg = len(self.imgs)
        self.npairs = len(self.imgs)

    def _read_image(self, fpath):
        try:
            return Image.open(fpath).convert('RGB')
        except Exception as e:
            raise IOError("Could not load image %s (reason: %s)" % (fpath, str(e)))
            sys.exit()

    def get_pair(self, pair_idx, output=()):
        fname1 = self.imgs[pair_idx]
        img1 = self._read_image(osp.join(self.path_style_img, fname1))

        # Pattern: 952.jpg.st_20170223_165410___winter_twilight.jpg
        orig_img, current_style = fname1.split(".st_")
        current_style = current_style.split("___")[1][:-4]

        # remove the used style from the self.styles_transfer_dict
        styles = list(self.styles_transfer_dict.keys())
        styles.remove(current_style)

        # let us randomly pick a new style
        style_2nd_img = random.choice(styles)
        if style_2nd_img == "orig":
            fname2 = osp.join(self.path_orig_img, orig_img)
        else:
            fname2 = osp.join(self.path_style_img,
                              orig_img + ".st_" + self.styles_transfer_dict[style_2nd_img] + "___" + style_2nd_img + ".jpg")

        img2 = self._read_image(fname2)

        W, H = img1.size
        sx = img2.size[0] / float(W)
        sy = img2.size[1] / float(H)

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


class AachenPairs_OpticalFlow (AachenImages_DB, PairDataset):
    """ Image pairs from Aachen db with optical flow.
    """
    def __init__(self, root='/ssd/data/r2d2_data/aachen/optical_flow', **kw):
        PairDataset.__init__(self)
        AachenImages_DB.__init__(self, **kw)
        self.root_flow = root

        # find out the subsest of valid pairs from the list of flow files
        flows = {f for f in os.listdir(os.path.join(root, 'flow')) if f.endswith('.png')}
        masks = {f for f in os.listdir(os.path.join(root, 'mask')) if f.endswith('.png')}
        assert flows == masks, 'Missing flow or mask pairs'
        
        make_pair = lambda f: tuple(self.db_image_idxs[v] for v in f[:-4].split('_'))
        self.image_pairs = [make_pair(f) for f in flows]
        self.npairs = len(self.image_pairs)
        assert self.nimg and self.npairs

    def get_mask_filename(self, pair_idx):
        tag_a, tag_b = map(self.get_tag, self.image_pairs[pair_idx])
        return os.path.join(self.root_flow, 'mask', f'{tag_a}_{tag_b}.png')

    def get_mask(self, pair_idx):
        return np.asarray(Image.open(self.get_mask_filename(pair_idx)))

    def get_flow_filename(self, pair_idx):
        tag_a, tag_b = map(self.get_tag, self.image_pairs[pair_idx])
        return os.path.join(self.root_flow, 'flow', f'{tag_a}_{tag_b}.png')

    def get_flow(self, pair_idx):
        fname = self.get_flow_filename(pair_idx)
        try:
            return self._png2flow(fname)
        except IOError:
            flow = open(fname[:-4], 'rb')
            help = np.fromfile(flow, np.float32, 1)
            assert help == 202021.25
            W, H = np.fromfile(flow, np.int32, 2)
            flow = np.fromfile(flow, np.float32).reshape((H, W, 2))
            return self._flow2png(flow, fname)

    def get_pair(self, idx, output=()):
        if isinstance(output, str): 
            output = output.split()

        img1, img2 = map(self.get_image, self.image_pairs[idx])
        meta = {}
        
        if 'flow' in output or 'aflow' in output:
            flow = self.get_flow(idx)
            assert flow.shape[:2] == img1.size[::-1]
            meta['flow'] = flow
            H, W = flow.shape[:2]
            meta['aflow'] = flow + np.mgrid[:H,:W][::-1].transpose(1,2,0)
        
        if 'mask' in output:
            mask = self.get_mask(idx)
            assert mask.shape[:2] == img1.size[::-1]
            meta['mask'] = mask
        
        return img1, img2, meta


if __name__ == '__main__':
    print(aachen_db_images)
    print(aachen_style_transfer_pairs)
    print(aachen_flow_pairs)
    pdb.set_trace()
