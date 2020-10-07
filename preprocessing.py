import os
import cv2
import numpy as np
from scipy import io
from tqdm import tqdm

BASE_PATH = ''

def _load_matfile(filepath):
    def _get_bbox(annot):
        _min = annot.min(axis=1)[:2, :]
        _max = annot.max(axis=1)[:2, :]
        bbox = np.concatenate([_min, _max], axis=0)
        bbox = bbox.T
        return bbox

    annotations = io.loadmat(filepath)['joints']
    result = {
        'bbox': _get_bbox(annotations)
    }
    return result

def save_cropped_image(bbox, image, save_path):
    cropped_path = 'cropped/images'
    os.makedirs(cropped_path, exist_ok=True)

    shape = image.shape

    def _bound_first(shape, bbox):
        min_x, min_y, max_x, max_y = bbox[0], bbox[1], bbox[2], bbox[3]
        buffer_length_x = max_x - min_x
        buffer_length_y = max_y - min_y
        upper_offset = 0
        lower_offset = 0
        left_offset = 0
        right_offset = 0
        min_x = min_x - buffer_length_x * 0.4
        max_x = max_x + buffer_length_x * 0.4
        min_y = min_y - buffer_length_y * 0.2
        max_y = max_y + buffer_length_y * 0.2
        if min_x < 0:
            left_offset = 0 - min_x
            min_x = 0
        if max_x > shape[1]:
            right_offset = max_x - shape[1]
            max_x = shape[1]
        if min_y < 0:
            upper_offset = 0 - min_y
            min_y = 0
        if max_y > shape[0]:
            lower_offset = max_y - shape[0]
            max_y = shape[0]
        offset = [upper_offset, lower_offset, left_offset, right_offset]
        point = [min_y, max_y, min_x, max_x]
        image_shape = [buffer_length_y * 1.4, buffer_length_x * 1.8]
        return offset, point, image_shape

    offset, point, image_shape = _bound_first(shape, bbox)
    round_offset = [int(round(num)) for num in offset]
    round_point = [int(round(num)) for num in point]
    round_image_shape = [int(round(num)) for num in image_shape]

    round_image_shape[0] = (round_point[1] - round_point[0]) + (round_offset[1]+round_offset[0])
    round_image_shape[1] = (round_point[3] - round_point[2]) + (round_offset[3]+round_offset[2])

    blank_image = np.zeros([round_image_shape[0], round_image_shape[1], 3])
    blank_image[
        round_offset[0]: round_image_shape[0] - round_offset[1],
        round_offset[2]: round_image_shape[1] - round_offset[3]
    ] = image[
        round_point[0]: round_point[1],
        round_point[2]: round_point[3]
    ]

    cv2.imwrite(save_path, blank_image)

def process_annotation():
    annotations = _load_matfile(os.path.join(BASE_PATH, 'joints.mat'))
    for i in tqdm(range(2000)):
        image = cv2.imread(
            os.path.join(BASE_PATH, 'images', 'im' + str(i+1).zfill(4) + '.jpg'))
        save_cropped_image(
            annotations['bbox'][i, :],
            image,
            os.path.join(BASE_PATH, 'cropped', 'images','im' + str(i+1).zfill(4) + '.jpg')
        )

if __name__ == "__main__":
    process_annotation()
