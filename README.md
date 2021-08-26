# Carla Pedestrian Detection
This pedestrian detection uses YOLO and cv2


 
## Steps:

1. Get this github repository in CARLA\WindowsNoEditor\PythonAPI\examples.
  It should look like this:
  
```
.CARLA_0.9.9            
├── WindowsNoEditor
│   │   ├── CarlaUE4
│   │   ├── Co-Simulation
│   │   ├── Engine
│   │   ├── HDMaps
│   │   ├── PythonAPI
│   │   │   ├── carla
│   │   │   ├── util
│   │   │   ├── examples
│   │   │   │ 	├── carla-pedestrian.py
│   │   │   │ 	├── coco.names
│   │   │   │ 	├── yoloyv4-tiny.cfg
│   │   │   │ 	├── yoloyv3-tiny.weights
          
```

2. Run spawn actor python file for  adding pedestrians or vehicles.

	```python spawn_npc.py```

3. Run pedestrian detection code

 	```python carla-pedestrian.py```


You should see two windows like this:
![Demo image](./image.png)

Control the car in the pygame window with WASD keys (this will be laggy, please be patient)
