# find_target_views.py
# we define one best view for each object present in the whole scene 

import os 
import json
import numpy as np

import ipdb

root_dir = 'avd/rohit_data'
image_dir = 'jpg_rgb'

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


with open('data/avd_instance_id_map.txt') as f: 
  lines = f.readlines()
  instance_num = len(lines)
  instance_names = ['' for _ in range(instance_num)]
  for line in lines:
    line = line.split()
    instance_names[int(line[1])] = line[0]

def find_target_views(subset, scene_list):
  json_path = 'data/avd_%s_tasks.json' % subset
  with open(json_path, 'w') as f:
    pass

  task_list = {}         

  for scene in scene_list:
    scene_dir = os.path.join(root_dir, scene)
    print scene_dir

    with open(os.path.join(scene_dir,'annotations.json')) as f:
      annotations = json.load(f)

    # with open(os.path.join(scene_dir,'present_instance_names.txt')) as f:
    #   instances = [l.strip('\n') for l in f.readlines()]
    instance_max_area = -np.ones(instance_num, dtype=np.float32)
    instance_min_diff = 100*np.ones(instance_num, dtype=np.int32)
    instance_max_image = ['' for _ in range(instance_num)]

    image_files = annotations.keys()
    for image_name in image_files:
      bboxes = annotations[image_name]['bounding_boxes']
      if len(bboxes) == 0: continue

      for bbox in bboxes: 
        instance_id = bbox[4] - 1
        diff_level = bbox[5]
        area = (bbox[2] - bbox[0]) * (bbox[3]-bbox[1])
        if area > instance_max_area[instance_id] and diff_level < instance_min_diff[instance_id]:
          instance_max_area[instance_id] = area 
          instance_max_image[instance_id] = image_name
          instance_min_diff[instance_id] = diff_level

    valid_image_names = []
    for i in range(instance_num):
      if instance_max_area[i] != -1 and instance_min_diff[i] <= 1:
        valid_image_names.append(instance_max_image[i])
    valid_image_names = set(valid_image_names)

    # NOTE: this is an ad-hoc solution (ideally we should store a name to id map in .h5)
    image_files = os.listdir(os.path.join(root_dir, scene, image_dir))
    task_name = 'avd_%s_%s'%(subset,scene)
    task_list[task_name] = []
    for image_name in valid_image_names:
      task_list[task_name].append(str(image_files.index(image_name)))

  with open(json_path, 'w') as f:
    json.dump(task_list, f)
    

if __name__ == '__main__':
  find_target_views('train', default_train_list)
  find_target_views('test', default_test_list)  
