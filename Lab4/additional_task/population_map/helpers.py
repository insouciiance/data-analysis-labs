import matplotlib.pyplot as plt
import numpy
from scipy.spatial import distance

def show_image(map_image, coords : list, values : list):
    _, axes = plt.subplots(figsize=(10, 10))

    axes.imshow(map_image)
    axes.axis('off')
    axes.scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        s=values,
        c='blue',
        alpha=0.4)

    plt.show()

def log_max_distance_info(map_image, coords : list, names : list):
    distances = distance.cdist(coords, coords, 'euclidean')
    first_city, second_city = numpy.unravel_index(distances.argmax(), distances.shape)

    distance_px = distances[first_city, second_city]

    ukraine_width_km = 1316
    ukraine_width_px = map_image.shape[1]
    km_per_pixel = ukraine_width_km / ukraine_width_px
    distance_km = distance_px * km_per_pixel

    print(f'{names[first_city]} and {names[second_city]} are the most distant.')
    print(f'Distance in px: {distance_px:.2f}')
    print(f'Distance in km: {distance_km:.2f}')
