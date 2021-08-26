#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.
Controls:
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import cv2
import time
import numpy as np
import tensorflow_yolov3.carla.utils as utils

import tensorflow as tf
from PIL import Image


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

import weakref
import random

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920//2
VIEW_HEIGHT = 1080//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)

# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None

        self.display = None
        self.image = None
        self.raw_image = None
        self.capture = True

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        #camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        #First person view transform settings
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=0))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
        

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.raw_image = cv2.cvtColor(array,cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))


    def pedestrian_detection(self, image, model, layer_name):
    
        NMS_THRESHOLD = 0.3
        MIN_CONFIDENCE = 0.2
        (H, W) = image.shape[:2]
        results = []
        personidz = 0 
        # constructu blob and this will retirn the bounding boxes and confidence values
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
        model.setInput(blob)
        layerOutputs = model.forward(layer_name)

        boxes = []
        centroids = []
        confidences = []

        # LayerOutputs is a list of lists containing outputs. Each list in layer output contains details about single prediction like its bounding box confidence 
        for output in layerOutputs:
            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if classID == personidz and confidence > MIN_CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
        # ensure at least one detection exists
        if len(idzs) > 0:
            # loop over the indexes we are keeping
            for i in idzs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # update our results list to consist of the person
                # prediction probability, bounding box coordinates,
                # and the centroid
                res = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(res)
        # return the list of results
        return results
    
    def game_loop(self):
        """
        Main program loop.
        """
        
        try:
            pygame.init()
            
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            
            input_size      = 416

            labelsPath = "coco.names"
            LABELS = open(labelsPath).read().strip().split("\n")

            weights_path = "yolov4-tiny.weights"
            config_path = "yolov4-tiny.cfg"

            model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            '''
            model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            '''

            layer_name = model.getLayerNames()
            layer_name = [layer_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]
            writer = None

            
            while True:
                
                self.world.tick()
    
                self.capture = True
                pygame_clock.tick_busy_loop(20)
    
                self.render(self.display)
                self.raw_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
                frame_size = self.raw_image.shape[:2]
                image_data = utils.image_preporcess(np.copy(self.raw_image), [input_size, input_size])
                image_data = image_data[np.newaxis, ...]

                ##here is more code ###
                  
                   
                results = self.pedestrian_detection(self.raw_image, model, layer_name)


                for res in results:
                    cv2.rectangle(self.raw_image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

                cv2.imshow("Detection", self.raw_image)
                    # utils.draw_bounding_boxes(pygame, self.display,  self.raw_image, results)
                # press esc key to stop 
                key = cv2.waitKey(1)
                if key == 27:
                    break
        
                pygame.display.flip()
    
                pygame.event.pump()
                if self.control(self.car):
                    return

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = BasicSynchronousClient()
        client.game_loop( )
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
