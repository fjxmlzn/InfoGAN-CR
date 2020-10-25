import numpy as np
from PIL import Image, ImageDraw

import math
import os


def generate_sample(theta, rad):
  image = Image.new(mode='L', size=(64, 64), color=0)
  draw = ImageDraw.Draw(image)
  x, y = center[0]+rad*math.cos(theta), center[1]+rad*math.sin(theta)
  draw.ellipse((x-r, y-r, x+r, y+r), fill=255)
  return image


# Parameters of CdSprites dataset
nangles = 40  # number of angles
angles = [float(tt)/nangles*math.pi*2 for tt in range(nangles)]
nradius = 27 # number of radii
radii = list(range(nradius))
center = (64//2 ,64//2) # center co-ordinate
r = 5 # radius of sprite

directory = '.'
if not os.path.exists(directory):
  os.makedirs(directory)
dataset_directory = directory
if not os.path.exists(dataset_directory):
  os.makedirs(dataset_directory)


# Generate the dataset
def generate_dataset():
  imgs = []
  latents_values = []

  for theta in angles:
    for rad in radii:
      image = generate_sample(theta, rad)
      imgs.append(np.array(image)//255)
      latents_values.append([theta, rad])

  metadata = {
    'description': 'Circular Disentanglement Sprites dataset (inspired from dSprites dataset by lmatthey@google.com). '
                   'Procedurally generated 2D shapes, from 2 positional disentangled latent factors: angle and radius of sprite. '
                   'All possible variations of the latents are present. '
                   , 
    'author': 'thekump2@illinois.edu', 
    'title': 'Circular dSprites (CdSprites) dataset',
    'latents_sizes': np.array([len(angles), len(radii)]),
    'latents_possible_values': {'angle': np.array(angles), 'radius': np.array(radii),},
    'date': 'Feb 2019',
    'latents_names': ('angle', 'radius',),
  }

  outfile = os.path.join(dataset_directory, 'cdsprites_ndarray_64x64.npz')
  np.savez(outfile, imgs=np.array(imgs), latents_values=np.array(latents_values), metadata=metadata)


# Random validation of generated dataset
def validate():
  dataset = np.load(os.path.join(dataset_directory, 'cdsprites_ndarray_64x64.npz'))

  imgs = dataset['imgs']
  random_indices = np.random.randint(imgs.shape[0], size=(20,))

  save_dir = os.path.join(directory, 'samples')
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  for ii in random_indices:
    image = Image.fromarray(np.uint8(imgs[ii]*255))
    image.save(os.path.join(save_dir, 'cdsprites_{}.png'.format(ii)))

  print(np.unique(imgs[random_indices], return_counts=True))
  print(np.unique((imgs[random_indices]*255), return_counts=True))
  print(np.unique(np.uint8(imgs[random_indices]*255), return_counts=True))

  print(np.unique(np.array(Image.fromarray(np.uint8(imgs[random_indices[0]]*255))), return_counts=True))
  print(np.unique(np.array(Image.fromarray(np.uint8(imgs[random_indices[0]]))), return_counts=True))


# Real Latent traversal
def latent_traversal():
  save_dir = os.path.join(directory, 'latent_traversals')
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  num_row, num_col = 10, 10
  height, width = 64, 64

  # Angle traversal
  image = np.zeros((num_row * height + (num_row-1),
                    num_col * width + (num_col-1),), dtype=np.uint8)
  for col, radius in enumerate(np.random.choice(radii, size=num_col, replace=False)):
    for row in range(num_row):
      angle = angles[row*(len(angles)//num_row)]
      image[row * (height+1): row * (height+1) + height,
            col * (width+1): col * (width+1) + width] = np.array(
              generate_sample(angle, radius))
  # Adding white lines
  image[height:(num_row-1)*(height+1):height+1, :] = 255
  image[:, width:(num_col-1)*(width+1):width+1] = 255

  image = Image.fromarray(image)
  image.save(os.path.join(save_dir, 'latent_traversal_angles.png'))


  # Radius traversal
  image = np.zeros((num_row * height + (num_row-1),
                    num_col * width + (num_col-1),), dtype=np.uint8)
  for col, angle in enumerate(np.random.choice(angles, size=num_col, replace=False)):
    for row in range(num_row):
      radius = radii[row*(len(radii)//num_row)]
      image[row * (height+1): row * (height+1) + height,
            col * (width+1): col * (width+1) + width] = np.array(
              generate_sample(angle, radius))
  # Adding white lines
  image[height:(num_row-1)*(height+1):height+1, :] = 255
  image[:, width:(num_col-1)*(width+1):width+1] = 255

  image = Image.fromarray(image)
  image.save(os.path.join(save_dir, 'latent_traversal_radius.png'))


# SHADE Real Latent traversal
def shade_latent_traversal():
  save_dir = os.path.join(directory, 'shade_latent_traversals')
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  num_row, num_col = 10, 10
  height, width = 64, 64

  # Angle traversal
  image = np.zeros((height,
                    num_col * width + (num_col-1),), dtype=np.uint8)
  for col, radius in enumerate(np.random.choice(radii, size=num_col, replace=False)):
    for row in range(num_row):
      angle = angles[row*(len(angles)//num_row)]
      image[:, col * (width+1): col * (width+1) + width] = np.maximum(
        (row*255//num_row)*(np.array(generate_sample(angle, radius))//255),
        image[:,col * (width+1): col * (width+1) + width])
  # Adding white lines
  image[:, width:(num_col-1)*(width+1):width+1] = 255

  image = Image.fromarray(image)
  image.save(os.path.join(save_dir, 'shade_latent_traversal_angles.png'))


  # Radius traversal
  image = np.zeros((height,
                    num_col * width + (num_col-1),), dtype=np.uint8)
  for col, angle in enumerate(np.random.choice(angles, size=num_col, replace=False)):
    for row in range(num_row):
      radius = radii[row*(len(radii)//num_row)]
      image[:, col * (width+1): col * (width+1) + width] = np.maximum(
        (row*255//num_row)*(np.array(generate_sample(angle, radius))//255),
        image[:,col * (width+1): col * (width+1) + width])

  # Adding white lines
  image[:, width:(num_col-1)*(width+1):width+1] = 255

  image = Image.fromarray(image)
  image.save(os.path.join(save_dir, 'shade_latent_traversal_radius.png'))


# Overlap of all images
def plot_overlap():
  save_dir = os.path.join(directory, '.')
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  num_row, num_col = 10, 10
  height, width = 64, 64

  # Angle traversal
  image = np.zeros((height, width), dtype=np.uint8)
  for radius in radii:
    for angle in angles:
      image[:, :] = np.maximum(
        np.array(generate_sample(angle, radius)),
        image[:, :])

  image = Image.fromarray(image)
  image.save(os.path.join(save_dir, 'overlap_{}angles_{}radius.png'.format(nangles, nradius)))


# Enumerate
def enumerate_dataset():
  save_dir = os.path.join(directory, 'enumerate')
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  height, width = 64, 64

  for radius in radii:
    for angle_idx, angle in enumerate(angles):
      image = np.zeros((height, width), dtype=np.uint8)
      image[:, :] = np.maximum(
        np.array(generate_sample(angle, radius)),
        image[:, :])
      image = Image.fromarray(image)
      image.save(os.path.join(save_dir, 'enumerate_{}angle_{}radius.png'.format(angle_idx, radius)))

if __name__ == '__main__':
  generate_dataset()
  validate()
  latent_traversal()
  shade_latent_traversal()
  plot_overlap()
  enumerate_dataset()