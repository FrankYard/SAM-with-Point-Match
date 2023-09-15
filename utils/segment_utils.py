import numpy as np
from typing import List, Tuple
import torch
from torch import Tensor
from sklearn_extra.cluster import KMedoids

def resize(image, self_min_size, self_max_size):
  # type: (Tensor, float, float) -> Tuple[Tensor]
  im_shape = torch.tensor(image.shape[-2:])
  min_size = float(torch.min(im_shape))
  max_size = float(torch.max(im_shape))
  scale_factor = self_min_size / min_size
  if max_size * scale_factor > self_max_size:
    scale_factor = self_max_size / max_size
  image = torch.nn.functional.interpolate(
      image[None], scale_factor=scale_factor, mode='bilinear', recompute_scale_factor=True,
      align_corners=False)[0]

  return image

def unify_size(size_list):
  v, _ = size_list.max(0)

  max_H, max_W = v[0].item(), v[1].item()
  
  new_H = (1 + (max_H - 1) // 32) * 32
  new_W = (1 + (max_W - 1) // 32) * 32

  return (new_H, new_W)

def pad_images(image_list, new_size=None, to_stack=False):
  if new_size is None:
    image_sizes = [(img.shape[-2], img.shape[-1]) for img in image_list]
    image_sizes = torch.tensor(image_sizes)
    new_size = unify_size(image_sizes)
  
  new_images = []
  for i in range(len(image_list)):
    size = image_list[i].shape[-2:]
    padding_bottom = new_size[0] - size[0]
    padding_right = new_size[1] - size[1]
    new_images += [torch.nn.functional.pad(image_list[i], (0, padding_right, 0, padding_bottom))]
  
  if to_stack:
    new_images = torch.stack(new_images, 0)
  
  return new_images, new_size


def preprocess_data(batch, config):

  min_size, max_size = config['normal_size']

  # resize images
  images = batch['image']
  new_images = []
  original_sizes = []
  new_sizes = []
  for i in range(len(images)):
    image = images[i]
    original_size = [image.shape[-2], image.shape[-1]]
    original_sizes.append(original_size)
    image = resize(image, min_size, max_size)
    new_size = [image.shape[-2], image.shape[-1]]
    new_sizes.append(new_size)
    new_images.append(image)
  images = new_images

  # batch data
  images, new_size = pad_images(images, to_stack=True)
  original_sizes = torch.tensor(original_sizes)
  new_sizes = torch.tensor(new_sizes)
  sizes = {'original_sizes':original_sizes, 'new_sizes':new_sizes, 'unified_new_size': new_size}
  assert images.shape[-2:] == new_size
  return images, sizes

def remove_overlap(scores, masks : List[np.ndarray], instances):
    s_out, m_out, i_out = [], [], []
    mask_union = np.zeros_like(masks[0])
    assert mask_union.dtype == bool
    for s, m, i in sorted(zip(scores, masks, instances), key=lambda x : x[0], reverse=True):
        overlap = mask_union * m
        if overlap.sum() > 0:
            m ^= overlap
            if m.max() == 0:
               continue

        mask_union += m
        s_out.append(s)
        m_out.append(m)
        i_out.append(i)
    return s_out, m_out, i_out
    
def sample_points(mask, select_num, subsample_size=1800):
    assert len(mask.shape) == 2
    mask_pixels = torch.from_numpy(mask).nonzero().float()
    if len(mask_pixels) < select_num:
        selected_points = mask_pixels.repeat(select_num // len(mask_pixels) + 1, 1)[:select_num]
    else:
        # Sample N points from the largest cluster by performing K-Medoids with K=N
        mask_pixels = mask_pixels[torch.randperm(len(mask_pixels))[:subsample_size]]
        selected_points = KMedoids(n_clusters=select_num).fit(mask_pixels).cluster_centers_
        selected_points = torch.from_numpy(selected_points).type(torch.float32)

    selected_points = selected_points.flip(1) # (y, x) -> (x, y)
    return selected_points

def get_crop_box(boxes, crop_scale_factor, box_padding, crop_normal, hw):
    assert len(hw) == 2
    crop_toplefts, crop_scales = [], []
    for box in boxes:
        assert box.dtype == int
        box_center = (box[[1,0]] + box[[3,2]]) // 2 #(y,x)
        crop_scale = (box[[3,2]] - box[[1,0]]) * crop_scale_factor + box_padding # h, w
        crop_scale = np.minimum(crop_scale, hw)
        # crop_scale = np.stack([crop_scale, box_center * 2 - 1, (image.shape[-2:]-box_center) * 2 - 1]).min(axis=0)
        hwratio_n = crop_normal[0] / crop_normal[1]
        hwratio = crop_scale[0] / crop_scale[1]
        if hwratio_n < hwratio:
            crop_scale[0] *= (hwratio_n/ hwratio)
        else:
            crop_scale[1] *= (hwratio / hwratio_n)
        crop_scale = crop_scale.astype(int)

        crop_topleft = box_center - (crop_scale//2)
        crop_topleft = np.minimum(np.maximum(crop_topleft, (0,0)), hw - crop_scale)
        crop_toplefts.append(crop_topleft)
        crop_scales.append(crop_scale)
    return crop_toplefts, crop_scales

def get_mask_img(masks, labels, out=None):
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    if isinstance(labels, list):
        labels = np.array(labels)
    colors = (labels[:, None] + 1) * palette
    colors = (colors % 255).astype("uint8")

    h, w = masks[0].shape
    if out is None:
        out = np.zeros((h, w, 3), dtype=np.uint8)
    for mask, color in zip(masks, colors):
        out[mask] = color

    return out