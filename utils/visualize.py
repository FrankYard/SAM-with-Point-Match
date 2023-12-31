import os
import cv2
import copy
import torch
import numpy as np

### write result to image
def compute_colors_for_labels(labels):
  """
  Simple function that adds fixed colors depending on the class
  """
  if type(labels) == torch.Tensor:
    labels = labels.numpy()
  elif type(labels) == np.ndarray:
    pass
  elif type(labels) == list:
    labels = np.array(labels)
  else:
    raise NotImplementedError
  
  palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
  colors = labels[:, None] * palette
  colors = (colors % 255).astype("uint8")
  return colors

def generate_csv(labels, file='segmentation_mapping.csv'):
  record = 'name,red,green,blue,alpha,id\n'
  colors = compute_colors_for_labels(labels)
  for i, (b,g,r) in enumerate(colors):
    record += f'instance{i},{r},{g},{b},255,{i}\n'
  with open(file, 'w') as f:
    f.write(record)
    
def overlay_boxes(image, boxes, labels, scores):
  """
  Adds the predicted boxes on top of the image

  Arguments:
      image (np.ndarray): an image as returned by OpenCV
  """
  colors = compute_colors_for_labels(labels).tolist()

  for box, color, score in zip(boxes, colors, scores):
    if score <= 0:
      continue
    top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
    image = cv2.rectangle(
        image, tuple(top_left), tuple(bottom_right), tuple(color), 1
    )

  return image

def overlay_class_names(image, boxes, labels, scores, categories):
  """
  Adds detected class names and scores in the positions defined by the
  top-left corner of the predicted bounding box

  Arguments:
      image (np.ndarray): an image as returned by OpenCV
  """

  if categories is not None:
    labels = [categories[i] for i in labels]

  template = "{}: {:.2f}"
  for box, score, label in zip(boxes, scores, labels):
    if score <= 0:
      continue
    x, y = box[:2]
    # x = round((box[0] + box[2])/2)
    # y = round((box[1] + box[3])/2)
    # s = template.format(label.item(), score)
    s = "{}".format(label)
    cv2.putText(
        image, s, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )

  return image

def findContours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatiblity between versions
    3 and 4

    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith('4'):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith('3'):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            'cv2 must be either version 3 or 4 to call this method')

    return contours, hierarchy

def overlay_mask(image, masks, labels, scores):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    colors = compute_colors_for_labels(labels).tolist()
    for mask, color, score in zip(masks, colors, scores):
      if score <= 0:
        continue
      # import pdb; pdb.set_trace()
      mask = mask.astype(np.uint8)
      contours, hierarchy = findContours(
          mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
      )
      image = cv2.drawContours(image, contours, -1, color, 2)

    composite = image

    return composite

def overlay_objects(images, detections, categories):
  new_images = []
  for image, detection in zip(images, detections):
    boxes, masks, labels, scores = detection['boxes'], detection['masks'], detection['labels'], detection['scores']
    image = overlay_mask(image, masks, labels, scores)
    image = overlay_boxes(image, boxes, labels, scores)
    image = overlay_class_names(image, boxes, labels, scores, categories)
    new_images.append(image)

  return images

def overlay_points(images, points_output, color=(255,0,0)):
  new_images = []
  for image, points_o in zip(images, points_output):
    points = points_o['points'].cpu().numpy()
    if len(points) == 0:
      print("no points")
    for i in range(len(points)):
      x = points[i][1].astype(int)
      y = points[i][0].astype(int)
      if x < 0:
        continue
      cv2.circle(image, (x,y), 1, color, thickness=-1)
    new_images.append(image)
  return new_images

def save_images(images, image_names, save_dir):
  for image, image_name in zip(images, image_names):
    save_path = os.path.join(save_dir, image_name)
    cv2.imwrite(save_path, image)

def save_detection_results(images, image_names, save_root=None, detections=None, categories=None, points_output=None, 
    save_objects=False, save_points=False):
  results = {}

  if save_root is not None:
    os.makedirs(save_root, exist_ok=True)

  if save_points and save_objects:
    images1 = copy.deepcopy(images)
    images1 = overlay_objects(images1, detections, categories)
    images1 = overlay_points(images1, points_output)
    results['points_and_objects'] = images1
    if save_root is not None:
      save_images(images1, image_names, save_root)

  elif save_objects:
    images1 = copy.deepcopy(images)
    images1 = overlay_objects(images1, detections, categories)
    results['objects'] = images1
    if save_root is not None:
      save_images(images1, image_names, save_root)
  
  elif save_points:
    images1 = copy.deepcopy(images)
    images1 = overlay_points(images1, points_output)
    results['points'] = images1
    if save_root is not None:
      save_images(images1, image_names, save_root)

  return results