# reorganize active vision dataset
#
# we choose to use hdf5 as backend
# we can also use memory as backend if we like 

from scipy import misc
import numpy as np
import json
from matplotlib import pyplot as plt

import ipdb


class World:
    def __init__(self, scenes, source=None, target=None):
        print 'Initializing world for (%s)' % ','.join(scenes)

        self.scenes = []
        
        self.maps = []
        for i in range(len(scenes)):
            self.maps.append(Map(scenes[i]))

        self.action_names = ["rotate_ccw","rotate_cw","forward","backward","left","right"]
        self.action_space = range(len(self.action_names))

        # NOTE: we can pass as an argument later 
        self.window_size = 4

        self.reset(source, target)

    def reset(self, source, target):
        # TODO: check if source and target is connected
        if not source:
            random_map_id = np.random.randint(len(self.maps))
            self.source_map = random_map_id
            random_view_id = np.random.randint(len(self.maps[random_map_id].data.keys()))
            self.source_view = self.maps[random_map_id].data.keys()[random_view_id]
        else:
            self.source_map = np.argmax([source in self.maps[i].data.keys() for i in range(len(self.maps))])
            self.source_view = source
            
        if not target:
            random_map_id = np.random.randint(len(self.maps))
            self.target_map = random_map_id
            random_view_id = np.random.randint(len(self.maps[random_map_id].data.keys()))
            self.target_view = self.maps[random_map_id].data.keys()[random_view_id]
        else:
            self.target_map = np.argmax([target in self.maps[i].data.keys() for i in range(len(self.maps))])
            self.target_view = target

        # we don't support cross-map navigation for now 
        assert(self.source_map == self.target_map)

        self.current_map = self.source_map
        self.current_view = self.source_view

        print 'Source[image %s from scene %s]' % (self.source_view, self.maps[self.source_map].scene)
        print 'Target[image %s from scene %s]' % (self.target_view, self.maps[self.target_map].scene)

        # # Visualization of source and target
        # src_im = misc.imread('avd/rohit_data/%s/jpg_rgb/%s' % (self.maps[self.source_map].scene,self.source_view))
        # dst_im = misc.imread('avd/rohit_data/%s/jpg_rgb/%s' % (self.maps[self.target_map].scene,self.target_view))
        # plt.subplot(2,2,1)
        # plt.imshow(src_im)
        # plt.title('source image')
        # plt.subplot(2,2,3)
        # plt.plot(self.maps[self.current_map].data[self.source_view]['feature'])        
        # plt.title('source feature')
        # plt.subplot(2,2,2)
        # plt.imshow(dst_im)
        # plt.title('target image')
        # plt.subplot(2,2,4)
        # plt.plot(self.maps[self.target_map].data[self.target_view]['feature'])
        # plt.title('target feature')
        # plt.show()
        # ipdb.set_trace()
               
    def step(self, action):
        assert not self.terminal, 'step() should not be called after termination'
        action_name = self.action_names[action]
        if action_name in self.maps[self.current_map].data[self.current_view]:
            # TODO: we might want to change map here later 
            self.current_view = self.maps[self.current_map].data[self.current][action_name]
            if self.check_terminal():
                self.terminal = True
                self.collided = False
            else:
                self.terminal = False
                self.collided = False
        else:
            self.terminal = False
            self.collided = False
            
        self.reward = self._reward()
        
    def check_terminal(self):
        return self.terminals[self.current_view]
            
    def _reward(self):
        if self.terminal:
            return 10.0
        else:
            return -0.1 if self.collided else -0.01
    
        
class Map:
    def __init__(self, scene):
        print 'Initializing map for %s ...' % scene
        
        self.scene = scene
        
        with open('avd/rohit_data/%s/annotations.json' % scene) as data_file:
            self.data = json.load(data_file)
            
        with open('avd/deep_features/resnet50/%s.json' % scene) as feat_file:
            features = json.load(feat_file)
            
        for name in self.data.iterkeys():
            self.data[name]['feature'] = np.array(features[name])
