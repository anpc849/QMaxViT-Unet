import torch
import torch.utils.data as data
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import random
import os
import h5py
from torch.distributions import Normal, Independent
from PIL import Image

def random_rot_flip(image, label, edge_mask=False):
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    if edge_mask:
        edge_mask = np.rot90(edge_mask, k)
        edge_mask = np.flip(edge_mask, axis=axis).copy()

    return image, label, edge_mask


def random_rotate(image, label, edge_mask, cval):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0,
                           reshape=False, mode="constant", cval=cval)
    if edge_mask:
        edge_mask = ndimage.rotate(edge_mask, angle, order=0,reshape=False)  

    return image, label, edge_mask


class RandomGenerator(object):
    def __init__(self, output_size, is_edge_mask=False):
        self.output_size = output_size
        self.is_edge_mask = is_edge_mask

    def __call__(self, sample):
        if self.is_edge_mask:
            image, label, edge_mask = sample['image'], sample['label'], sample['edge_mask']
        
        else:
            image, label = sample['image'], sample['label']
       
        if random.random() > 0.5:
            image, label, edge_mask = random_rot_flip(image, label, edge_mask if self.is_edge_mask else False)
        elif random.random() > 0.5:
            if 4 in np.unique(label):
                image, label, edge_mask = random_rotate(image, label, edge_mask if self.is_edge_mask else False,cval=4)
            else:
                image, label, edge_mask = random_rotate(image, label, edge_mask if self.is_edge_mask else False,cval=0)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        
        if self.is_edge_mask:
            if not isinstance(edge_mask, np.ndarray):
                edge_mask = np.array(edge_mask)
            edge_mask = torch.from_numpy(
                edge_mask.astype(np.float32)).unsqueeze(0)
            sample = {'image': image, 'label': label, 'edge_mask': edge_mask}
            return sample
            
        sample = {'image': image, 'label': label}

        return sample

def generate_edge_mask(old_path_data, new_path_data, edge_detector):

    h5_files = os.listdir(old_path_data) ## contains h5 files of training slices

    for h5_file in h5_files:
        h5_file_name = os.path.join(old_path_data, h5_file)
        h5f = h5py.File(h5_file_name, 'r')
        image = h5f['image'][:]

        x,y = image.shape
        image = zoom(image, (256 / x, 256 / y), order=0)
        image = (image - image.min()) / (image.max() - image.min())


        gray_image_tensor = torch.tensor(min_max_image, dtype=torch.float32).unsqueeze(0)
        rgb_image_tensor = torch.cat((gray_image_tensor, gray_image_tensor, gray_image_tensor), dim=0)

        image_input = rgb_image_tensor.unsqueeze(0)
        image_input = image_input.cuda()

        with torch.no_grad():
            mean, std = edge_detector(image_input)  # Add batch dimension
            _,_,H,W = image_input.shape
            outputs_dist=Independent(Normal(loc=mean, scale=std+0.001), 1)
            outputs = torch.sigmoid(outputs_dist.rsample())
            edge_mask=torch.squeeze(outputs.detach()).cpu().numpy()

        sample = {'image': h5f['image'][:], 'label': h5f['scribble'][:], "edge_mask": edge_mask}

        output_file_one = new_path_data + "/MSCMR_training_slices/" + h5_file ## change the name dataset such as ACDC_training_slices

        with h5py.File(output_file_one, "w") as f:
            for key, value in sample.items():
                f.create_dataset(key, data=value)