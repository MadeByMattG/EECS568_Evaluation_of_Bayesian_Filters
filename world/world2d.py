import sys
sys.path.append('.')
import yaml
import numpy as np

from utils.Landmark import *

class world2d:

    def __init__(self):

        with open("config/settings.yaml", 'r') as stream:
            param = yaml.safe_load(stream)

        landmark_params = param['landmarks']

        self.num_landmarks_ = len(landmark_params)
        self.landmark_list_ = LandmarkList()
        self.marker_pos = np.zeros((self.num_landmarks_, 2))
        self._cubes = []  # (cube_id, center_pos, [1-indexed tag ids]) for marker display

        for entry in landmark_params:
            cube_id = entry['id']
            center = np.array([entry['x'], entry['y']])
            half = entry['size'] / 2.0
            yaw = entry['yaw']
            tag_ids_1indexed = []

            for i, raw_tag_id in enumerate(entry['tag_ids']):
                angle = yaw + i * (np.pi / 2.0)
                face_pos = center + half * np.array([np.cos(angle), np.sin(angle)])
                tag_id_1indexed = raw_tag_id + 1
                self.landmark_list_.addLandmark(Landmark(tag_id_1indexed, face_pos))
                tag_ids_1indexed.append(tag_id_1indexed)

            self._cubes.append((cube_id, center, tag_ids_1indexed))
            self.marker_pos[cube_id - 1, :] = center

    def getLandmarksInWorld(self):
        return self.landmark_list_

    def getLandmark(self, id):
        return self.landmark_list_.getLandmark(id)

    def getNumLandmarks(self):
        return self.num_landmarks_

    def getCubes(self):
        """Returns list of (cube_id, position, [1-indexed tag ids]) for marker display."""
        return self._cubes


def main():

    robot_world = world2d()

    pass

if __name__ == '__main__':
    main()
