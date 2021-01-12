import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import time
import functools


def connect_neighbors(sample, sample_region):
    # recursively add connected samples to the sample's region
    sample_region.add(sample.index)
    for connected_sample in sample.directly_connected_samples:
        if connected_sample.index not in sample_region:
            connect_neighbors(connected_sample, sample_region)


class SamplePoint:
    def __init__(self, position, index):
        self.position = position
        self.index = index

        self.directly_connected_samples = []
        self.region = None
        self.adjacent_surface_indices = []
        self.reduced_adjacent_surface_indices = []
        self.same_surfaces = None

    def __repr__(self):
        return 'S-Index: ' + str(self.index)

    def compare_surfaces(self):
        if set(self.adjacent_surface_indices) == set(self.reduced_adjacent_surface_indices):
            self.same_surfaces = True
        else:
            self.same_surfaces = False
        return

    def grow_connected_region(self):
        # region growing on connected samples
        sample_region = set()
        connect_neighbors(self, sample_region)
        self.region = sample_region


class Pixel:
    def __init__(self, position):
        self.position = position
        self.closest_sample = None
        self.dist_sample = None

        self.dist_surface = None
        self.reduced_dist_surface = None

        self.gray_value = None
        self.reduced_gray_value = None

    def __repr__(self):
        return 'Pixel: ' + str(self.position)

    def min_dist_sample(self, sample_points):
        min_dist = np.inf
        for i in range(len(sample_points)):
            sample_position = sample_points[i].position
            sample_index = sample_points[i].index
            dist = np.linalg.norm(sample_position - self.position)
            if dist < min_dist:
                min_index = sample_index
                min_dist = dist
        self.closest_sample = sample_points[min_index]
        self.dist_sample = min_dist
        return

    def pixel_plane_dist(self, plane):
        normal, d = plane
        dist = abs(np.sum(normal * self.position) + d) / np.linalg.norm(normal)
        return dist

    def min_dist_surface(self, surfaces):
        min_dist = np.inf
        # check if one of the samples surfaces is dropped
        if self.closest_sample.same_surfaces:
            # same result for both
            for surface_index in self.closest_sample.adjacent_surface_indices:
                dist = self.pixel_plane_dist(surfaces[surface_index])

                if dist < min_dist:
                    min_dist = dist

            self.dist_surface = min_dist
            self.reduced_dist_surface = min_dist

        else:
            # do both surface iterations separately
            for surface_index in self.closest_sample.adjacent_surface_indices:
                dist = self.pixel_plane_dist(surfaces[surface_index])

                if dist < min_dist:
                    min_dist = dist

            self.dist_surface = min_dist

            reduced_min_dist = np.inf
            # print(self.closest_sample.directly_connected_samples)
            # for connected_sample in self.closest_sample.directly_connected_samples:
            # TODO: iterate surfaces of all connected neighbors, make sure all neighbors are tracked for all samples within each group
            for surface_index in self.closest_sample.reduced_adjacent_surface_indices:
                dist = self.pixel_plane_dist(surfaces[surface_index])

                if dist < reduced_min_dist:
                    reduced_min_dist = dist

            self.reduced_dist_surface = reduced_min_dist
        return

    def compute_gray_value(self, rand_num):
        self.gray_value = 1 - np.power(self.dist_surface, 1/10) + rand_num
        self.reduced_gray_value = 1 - np.power(self.reduced_dist_surface, 1 / 10) + rand_num
        return


class IndexTracker(object):
    """
    Class to visualize the 3D images.
    Adapted from https://matplotlib.org/gallery/animation/image_slices_viewer.html

    """

    def __init__(self, ax, A, B, C, D, E, F):
        # A, B, C, D 3D images of shape image_width^3 stored in np.array
        self.ax = ax

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F
        rows, cols, self.slices = A.shape
        self.ind = self.slices//2

        self.im0 = ax[0, 0].imshow(self.A[:, :, self.ind])
        self.ax[0, 0].set_title('Original')
        self.im1 = ax[0, 1].imshow(self.B[:, :, self.ind], cmap='gray')
        self.ax[0, 1].set_title('Border')
        self.im2 = ax[0, 2].imshow(self.C[:, :, self.ind], cmap='gray')
        self.ax[0, 2].set_title('Reduced Border')
        self.im3 = ax[1, 0].imshow(self.D[:, :, self.ind], cmap='gray')
        self.ax[1, 0].set_title('Noise')
        self.im4 = ax[1, 1].imshow(self.E[:, :, self.ind], cmap='gray')
        self.ax[1, 1].set_title('Reduced Noise')
        self.im5 = ax[1, 2].imshow(self.F[:, :, self.ind])
        self.ax[1, 2].set_title('Reduced Original')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im0.set_data(self.A[:, :, self.ind])
        self.ax[0, 0].set_ylabel('slice %s' % self.ind)
        self.im1.set_data(self.B[:, :, self.ind])
        self.im2.set_data(self.C[:, :, self.ind])
        self.im3.set_data(self.D[:, :, self.ind])
        self.im4.set_data(self.E[:, :, self.ind])
        self.im5.set_data(self.F[:, :, self.ind])

        self.im0.axes.figure.canvas.draw()


def voronoi_diagram(num=1000):
    # sample a number of points in the unit cube
    samples = np.random.uniform(low=0, high=1, size=(num, 3))
    boundary_points = np.array([[-1, -1, -1], [2, -1, -1], [-1, 2, -1], [2, 2, -1],
                         [-1, -1, 2], [2, -1, 2], [-1, 2, 2], [2, 2, 2]])
    samples = np.vstack((samples, boundary_points))

    # create Voronoi diagram
    vor = Voronoi(samples)

    # create sample point objects
    sample_point_objects = []

    for point_index, point in enumerate(vor.points):
        p = SamplePoint(point, point_index)
        sample_point_objects.append(p)
    return vor, sample_point_objects


def build_surface_dict(vor):
    """
    Adapted from https://stackoverflow.com/questions/53698635/how-to-define-a-plane-with-3-points-and-plot-it-in-3d
    (Reblochon Masque)

    """
    # compute normal vectors and origin distances of planes from first 3 vertices
    surfaces = {}
    for surface_index, surface_vertex_indices in enumerate(vor.ridge_vertices):

        if not surface_vertex_indices:
            continue
        if -1 in surface_vertex_indices:
            continue

        x0, y0, z0 = vor.vertices[surface_vertex_indices[0]]
        x1, y1, z1 = vor.vertices[surface_vertex_indices[1]]
        x2, y2, z2 = vor.vertices[surface_vertex_indices[2]]

        ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]
        vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]

        u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]

        point = np.array(vor.vertices[surface_vertex_indices[0]])
        normal = np.array(u_cross_v)

        d = -point.dot(normal)
        surfaces[surface_index] = (normal, d)
    return surfaces


def draw_collection_lines(verts):
    # TODO: not all edges are drawn, verts not ordered?
    # draw the edge lines for a polygon given by the vertices
    for itr in range(0, len(verts)):
        plt.plot(
            [verts[itr - 1, 0], verts[itr, 0]],
            [verts[itr - 1, 1], verts[itr, 1]],
            [verts[itr - 1, 2], verts[itr, 2]],
            color="k"
        )
    return


def cartesian_product_broadcasted(*arrays):
    """
    http://stackoverflow.com/a/11146645/190597 (senderle)
    """
    broadcastable = np.ix_(*arrays)
    broadcasted = np.broadcast_arrays(*broadcastable)
    dtype = np.result_type(*arrays)
    rows, cols = functools.reduce(np.multiply, broadcasted[0].shape), len(broadcasted)
    out = np.empty(rows * cols, dtype=dtype)
    start, end = 0, rows
    for a in broadcasted:
        out[start:end] = a.reshape(-1)
        start, end = end, end + rows
    return out.reshape(cols, rows).T


def plot_voronoi_3d(image):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    size = image.shape[0]
    volume = image
    x, y, z = cartesian_product_broadcasted(*[np.arange(size, dtype='int16')] * 3).T
    mask = ((x == 0) | (x == size - 1)
            | (y == 0) | (y == size - 1)
            | (z == 0) | (z == size - 1))
    x = x[mask]
    y = y[mask]
    z = z[mask]
    volume = volume.ravel()[mask]

    ax.scatter(x, y, z, c=volume, cmap=plt.get_cmap('Greys'))
    plt.show()


def main():
    start_time = time.time()

    num_samples = 10
    num_removed_surfaces = 10
    image_width = 10  # change to adapt resolution

    # create Voronoi diagram
    vor, sample_points = voronoi_diagram(num_samples)

    surfaces = build_surface_dict(vor)

    # add adjacent surfaces to each sample point
    reduced_ridge_point_indices = np.random.choice(range(len(vor.ridge_points)), num_removed_surfaces)

    for surface_index, split_points in enumerate(vor.ridge_points):

        if -1 in vor.ridge_vertices[surface_index]:
            continue

        sample0 = sample_points[split_points[0]]
        sample1 = sample_points[split_points[1]]

        # add normal-distance pair of adjacent plane to sample point
        sample0.adjacent_surface_indices.append(surface_index)
        sample1.adjacent_surface_indices.append(surface_index)

        if surface_index not in reduced_ridge_point_indices:
            sample0.reduced_adjacent_surface_indices.append(surface_index)
            sample1.reduced_adjacent_surface_indices.append(surface_index)
        else:
            sample0.directly_connected_samples.append(sample1)
            sample1.directly_connected_samples.append(sample0)

    # remove surface duplicates and compare surface lists
    for sample_point in sample_points:
        sample_point.adjacent_surface_indices = list(set(sample_point.adjacent_surface_indices))
        sample_point.reduced_adjacent_surface_indices = list(set(sample_point.reduced_adjacent_surface_indices))
        sample_point.directly_connected_samples = list(set(sample_point.directly_connected_samples))

    for sample_point in sample_points:
        # directly_connected_samples have to be built before for all samples
        sample_point.grow_connected_region()

    # construct images
    image = np.zeros((image_width, image_width, image_width), dtype=int)
    reduced_image = np.zeros(image.shape, dtype=int)
    border = np.zeros(image.shape, dtype=int)
    reduced_border = np.zeros(image.shape, dtype=int)
    noise = np.zeros(image.shape)
    reduced_noise = np.zeros(image.shape)
    noise_generator = np.random.normal(0, 0.01, size=image.shape) * 1

    pixels = []  # list of length image_width^3

    for i in range(image_width):
        print('Progress: Layer', i)
        for j in range(image_width):
            for k in range(image_width):
                pixel = Pixel(np.array([i, j, k]) / image_width)
                pixel.min_dist_sample(sample_points=sample_points)
                pixels.append(pixel)

                image[i, j, k] = pixel.closest_sample.index
                reduced_image[i, j, k] = min(pixel.closest_sample.region)

                pixel.min_dist_surface(surfaces=surfaces)

                pixel.compute_gray_value(rand_num=noise_generator[i, j, k])

                if pixel.dist_surface < 0.01:
                    border[i, j, k] = 1

                if pixel.reduced_dist_surface < 0.01:
                    reduced_border[i, j, k] = 1

                noise[i, j, k] = pixel.gray_value
                reduced_noise[i, j, k] = pixel.reduced_gray_value

    # reformat image to 8bit
    reduced_noise = np.interp(reduced_noise, (reduced_noise.min(), reduced_noise.max()), (0, 255)).astype(np.uint8)

    np.save('../storage/image', image)
    np.save('../storage/image + reduced', reduced_image)
    np.save('../storage/border', border)
    np.save('../storage/border + reduced', reduced_border)
    np.save('../storage/noise', noise)
    np.save('../storage/noise + reduced', reduced_noise)

    print("--- %s seconds ---" % (time.time() - start_time))

    fig, ax = plt.subplots(2, 3)
    tracker = IndexTracker(ax, image, border, reduced_border, noise, reduced_noise, reduced_image)
    fig.suptitle('Use scroll wheel to navigate slices \nImage dimensions: ({}, {}, {}) \n'
                 'Number samples: {} \nNumber dropped surfaces: {}'
                 .format(image_width, image_width, image_width, num_samples, num_removed_surfaces))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

    plot_voronoi_3d(image)
    plot_voronoi_3d(reduced_image)

    return image


if __name__ == "__main__":
    main()
