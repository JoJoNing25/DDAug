from scipy.ndimage.measurements import center_of_mass
from PIL import Image
import numpy as np


class Point():
    """
    Point Class For X and Y Location
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y


def calc_tile_locations(tile_size, image_size):
    """
    Divide an image into tiles to help us cover classes that are spread out.
    tile_size: size of tile to distribute
    image_size: original image size
    return: locations of the tiles
    """
    image_size_y, image_size_x = image_size
    locations = []
    for y in range(image_size_y // tile_size):
        for x in range(image_size_x // tile_size):
            x_offs = x * tile_size
            y_offs = y * tile_size
            locations.append((x_offs, y_offs))
    return locations


def class_centroids_image(label_fn, tile_size=1024, num_classes=35):
    """
    For one image, calculate centroids for all classes present in image.
    item: image, image_name
    tile_size:
    num_classes:
    id2trainid: mapping from original id to training ids
    return: Centroids are calculated for each tile.
    """
    centroids = []
    mask = np.array(Image.open(label_fn))
    image_size = mask.shape
    tile_locations = calc_tile_locations(tile_size, image_size)
    for x_offs, y_offs in tile_locations:
        patch = mask[y_offs:y_offs + tile_size, x_offs:x_offs + tile_size]
        for class_id in range(num_classes):
            if class_id in patch:
                patch_class = (patch == class_id).astype(int)
                centroid_y, centroid_x = center_of_mass(patch_class)
                centroid_y = int(centroid_y) + y_offs
                centroid_x = int(centroid_x) + x_offs
                centroid = (centroid_x, centroid_y)
                centroids.append(centroid)
    return centroids
