# NOTE: convert_avd_to_thor.py
# convert AVD to THOR format 
# generate a similar hdf5 file for each scene 

import os
import math 
import json 
import h5py 
from scipy import misc 
import scipy.io as sio
import numpy as np 
import tensorflow as tf
from matplotlib import pyplot as plt 

import ipdb 

# train_set = active_vision_dataset.AVD(root='avd/rohit_data')
default_train_list = ['Home_02_1',
                      'Home_03_1',
                      'Home_03_2',
                      'Home_04_1',
                      'Home_04_2',
                      'Home_05_1',
                      'Home_05_2',
                      'Home_06_1',
                      'Home_14_1',
                      'Home_14_2',
                      'Office_01_1']
default_test_list = ['Home_01_1',
                     'Home_01_2',
                     'Home_08_1']
root_dir = 'avd/rohit_data'
feat_dir = 'avd/deep_features/resnet50'
image_dir = 'jpg_rgb'
action_list = ['rotate_ccw', 'rotate_cw', 'forward', 'backward', 'left', 'right']

def image_to_scene(image_name):
  if image_name[0] == '0':
    scene_type = 'Home'
  else:
    scene_type = 'Office'
  scene_name = scene_type + '_' + image_name[2:4] + '_' + image_name
  return scene_name

def convert_avd_to_thor(subset, scene_list):
  print 'Converting %s subset' % subset
  for scene in scene_list:
    print 'processing scene %s' % (scene)
    image_files = os.listdir(os.path.join(root_dir, scene, image_dir))

    # load object and transition annotation
    scene_dir = os.path.join(root_dir, scene)
    with open(os.path.join(scene_dir,'annotations.json')) as f:
      annotations = json.load(f)

    # load camera annotation
    image_structs_path = os.path.join(scene_dir, 'image_structs.mat')
    image_structs = sio.loadmat(image_structs_path)
    cam_scale = image_structs['scale'][0][0]
    image_structs = image_structs['image_structs']
    image_structs = image_structs[0]
    cam_world_pos = {}
    cam_direction = {}
    for (i, cam) in enumerate(image_structs):
      image_name = cam[0][0]
      cam_world_pos[image_name] = cam[3]
      cam_direction[image_name] = cam[4]

    # load resnet features 
    with open(os.path.join(feat_dir, '%s.json' % scene)) as feat_file: 
      image_features = json.load(feat_file)

    graph = list()
    location = list()
    rotation = list()
    observation = list()
    resnet_feature = list()

    # enumerate image files 
    for (i, image_name) in enumerate(image_files): 
      print 'processing images (%d/%d) %s' % (i, len(image_files), image_name)
      # define graph
      next_image_indices = np.array([-1 for _ in action_list])
      for (j, action) in enumerate(action_list):
        next_image_name = annotations[image_name][action]
        next_image_indices[j] = -1 if next_image_name == '' else image_files.index(next_image_name)
      graph.append(next_image_indices)

      # define location (ignore height since they are the same)
      cam_xz = cam_world_pos[image_name]
      cam_xz = np.array([cam_xz[0][0], cam_xz[2][0]])
      location.append(cam_xz)

      # define rotation
      dir_vec = cam_direction[image_name]
      rot_ang = math.atan2(dir_vec[2], dir_vec[0]) / math.pi * 180
      rotation.append(rot_ang)

      # define observations
      im = misc.imread(os.path.join(root_dir, scene, image_dir, image_name))
      im = misc.imresize(im, (224,224))
      im = np.array(im)
      observation.append(im)

      # define resnet features 
      feat_vec = image_features[image_name]
      resnet_feature.append(feat_vec)

    graph = np.array(graph,dtype=np.int32)
    location = np.array(location,dtype=np.float32)
    rotation = np.array(rotation,dtype=np.float32)
    observation = np.array(observation,dtype=np.int8)
    resnet_feature = np.array(resnet_feature,dtype=np.float32)
    # NOTE: to mimic what THOR does (n_feat_per_loc)
    resnet_feature = np.expand_dims(resnet_feature, axis=1)

    # initialize hdf5 format
    f = h5py.File('data/avd_%s_%s.h5'%(subset, scene), 'w')
    f.create_dataset('graph', data=graph)
    f.create_dataset('location', data=location)
    f.create_dataset('rotation', data=rotation)
    f.create_dataset('observation', data=observation)
    f.create_dataset('resnet_feature', data=resnet_feature)
    f.close()

if __name__ == '__main__':
  convert_avd_to_thor('train', default_train_list)
  convert_avd_to_thor('test', default_test_list)
