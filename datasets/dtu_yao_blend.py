from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
from datasets.preprocess import *

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, inverse_depth=False,
                 origin_size=False, light_idx=-1, image_scale=0.25, reverse=False, both=True, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.inverse_depth = inverse_depth
        self.origin_size = origin_size
        self.light_idx=light_idx
        self.image_scale = image_scale # use to resize image
        self.reverse = reverse
        self.both = both
        print('dataset: inverse_depth {}, origin_size {}, light_idx:{}, image_scale:{}, reverse: {}, both: {}'.format(
                    self.inverse_depth, self.origin_size, self.light_idx, self.image_scale, self.reverse, self.both))
        
        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "{}/cams/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) < self.nviews -1:
                        print('less ref_view small {}'.format(self.nviews-1))
                        continue
                    if self.both:
                        metas.append((scan, ref_view, src_views, 1)) # add 1, 0 for reverse depth
                    metas.append((scan, ref_view, src_views, 0))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        if self.image_scale != 1.0: # origin: 1.0
            intrinsics[:2, :] *= self.image_scale

        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        if self.image_scale != 1.0:
            w, h = img.size
            img = img.resize((int(self.image_scale * w), int(self.image_scale*h))) # origin: 0.25
        # scale 0~255 to 0~1
        #np_img = np.array(img, dtype=np.float32) / 255. # origin version on 2020/02/20
        #return np_img
        return self.center_img(np.array(img, dtype=np.float32))	

    def center_img(self, img): # this is very important for batch normalization
        img = img.astype(np.float32)
        var = np.var(img, axis=(0,1), keepdims=True)
        mean = np.mean(img, axis=(0,1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + 0.00000001)

    def read_depth(self, filename):
        # read pfm depth file
        depth_image = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_image = scale_image(depth_image, scale=self.image_scale, interpolation='nearest')
        return depth_image
        

    def __getitem__(self, idx):
        
        #print('idx: {}, flip_falg {}'.format(idx, flip_flag))
        meta = self.metas[idx]
        scan, ref_view, src_views, flip_flag = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 000000000
            img_filename = os.path.join(self.datapath,
                                        '{}/blended_images/{:0>8}.jpg'.format(scan, vid)) 
            # if i == 0:
            #     print('process in {}, {}'.format(idx, img_filename))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))
            
            if i == 0:
                depth_name = depth_filename
            #print('debug in dtu_yao', i, depth_filename)
            imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                if self.inverse_depth: #slice inverse depth
                    print('inverse depth')
                    depth_end = depth_interval * (self.ndepths-1) + depth_min # Origin version: depth_interval * n-1 + depth_min as the last clos
                    depth_values = np.linspace(1.0 / depth_min, 1.0/depth_end , self.ndepths, endpoint=False)
                    depth_values = 1.0 / depth_values
                    depth_values = depth_values.astype(np.float32)
                    
                    #depth_values = np.concatenate((depth_values[1::2],depth_values[::-2]),axis=0)
                    
                else:
                    depth_values = np.arange(depth_min, depth_interval * (self.ndepths-0.5) + depth_min, depth_interval,
                                            dtype=np.float32) # the set is [)
                                            
                    depth_values = np.concatenate((depth_values,depth_values[::-1]),axis=0)
                    
                    depth_end = depth_interval * (self.ndepths-1) + depth_min
                
                #mask = self.read_img(mask_filename)
                depth = self.read_depth(depth_filename)
                #mask = np.array((depth > depth_min+depth_interval) & (depth < depth_min+(self.ndepths-2)*depth_interval), dtype=np.float32)
                mask = np.array((depth >= depth_min) & (depth <= depth_end), dtype=np.float32)
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        if (flip_flag and self.both) or (self.reverse and not self.both):
            depth_values = np.array([depth_values[len(depth_values)-i-1]for i in range(len(depth_values))])
        
        #print('img:{}, depth:{}, depth_values:{}, mask:{}, depth_interval:{}'.format(imgs.shape, depth.shape, depth_values.shape,mask.shape,depth_interval))
        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "depth": depth,
                "depth_values": depth_values, # generate depth index
                "mask": mask,
                "depth_interval": depth_interval,
                'name':depth_name,}
