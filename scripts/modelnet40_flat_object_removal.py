# Script to remove specific object files from the dataset
# These objects might cause issues during pointnet training 
# so it is safe to remove them first

import os

pointnet40_path = 'data/modelnet40/PointClouds'
paths_to_remove = {
    'plant': [
        '/plant/train/plant_0232.npy',
        '/plant/train/plant_0044.npy',
        '/plant/test/plant_0265.npy',
        '/plant/test/plant_0276.npy'
    ],
    'guitar': [
        '/guitar/train/guitar_0036.npy',
        '/guitar/train/guitar_0147.npy'
    ],
    'person': [
        '/person/train/person_0051.npy',
        '/person/train/person_0026.npy',
        '/person/train/person_0030.npy',
        '/person/train/person_0044.npy',
        '/person/train/person_0058.npy',
        '/person/test/person_0101.npy',
        '/person/test/person_0093.npy'
    ],
    'door': [
        '/door/train/door_0059.npy'
    ]
}

for category, files in paths_to_remove.items():
    for file_path in files:
        file_path = pointnet40_path + file_path
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed {file_path}")
        else:
            print(f"File not found: {file_path}")
