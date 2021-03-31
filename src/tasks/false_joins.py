import numpy as np
from scipy.spatial import Voronoi
import time
import os
import matplotlib.pyplot as plt
from scipy import ndimage

from ..utils import IndexTracker, plot_3d


class TriangleSamples:
    # https: // stackoverflow.com / questions / 11178414 / algorithm - to - generate - equally - distributed - points - in -a - polygon
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

        self.area = np.linalg.norm(np.cross(b - a, c - a)) / 2

    def get_triangle_samples(self, image_width):
        num_triangle_samples = int(self.area * image_width ** 2)

        r = np.random.uniform(size=(num_triangle_samples, 1))
        d = (1 - r) * self.a + r * self.b
        s = np.random.uniform(size=(num_triangle_samples, 1))
        e = (1 - np.sqrt(s)) * self.c + np.sqrt(s) * d

        # e is uniformly sampled across the surface
        rounded_sample_points = np.around(e[(e.min(axis=1) >= 0) & (e.max(axis=1) <= 1), :] * image_width).astype(int)
        rounded_sample_points = np.unique(rounded_sample_points[rounded_sample_points.max(axis=1) < image_width], axis=0)  # remove index image_width
        return rounded_sample_points


class Edge:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.delta = b - a
        self.length_squared = np.linalg.norm(self.delta) ** 2

    def point_at(self, t):
        return self.a + t * self.delta

    def project(self, p):
        return np.dot((p - self.a), self.delta) / self.length_squared


class Plane:
    def __init__(self, point, direction):
        self.point = point
        self.direction = direction

    def is_above(self, q):
        return np.dot(self.direction, (q - self.point)) > 0


class Triangle:
    def __init__(self, a, b, c):
        self.edge_ab = Edge(a, b)
        self.edge_bc = Edge(b, c)
        self.edge_ca = Edge(c, a)
        self.trinorm = np.cross(a-b, a-c)
        self.vertices = [a, b, c]

        self.triplane = Plane(a, self.trinorm)

        self.plane_ab = Plane(a, np.cross(self.trinorm, self.edge_ab.delta))
        self.plane_bc = Plane(b, np.cross(self.trinorm, self.edge_bc.delta))
        self.plane_ca = Plane(c, np.cross(self.trinorm, self.edge_ca.delta))

    def clostest_point_to(self, p):
        uab = self.edge_ab.project(p)
        uca = self.edge_ca.project(p)

        if uca > 1 and uab < 0:
            return self.edge_ab.a

        ubc = self.edge_bc.project(p)

        if uab > 1 and ubc < 0:
            return self.edge_bc.a
        if ubc > 1 and uca < 0:
            return self.edge_ca.a
        if 0 < uab < 1 and not self.plane_ab.is_above(p):
            return self.edge_ab.point_at(uab)
        if 0 < ubc < 1 and not self.plane_bc.is_above(p):
            return self.edge_bc.point_at(ubc)
        if 0 < uca < 1 and not self.plane_ca.is_above(p):
            return self.edge_ca.point_at(uca)

        # clostest triangle inside triangle
        return None


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

    def grow_connected_region(self, sample_points):
        # region growing on connected samples
        sample_region = set()
        connect_neighbors(self, sample_region)
        self.region = sample_region

        # combine surface indices of all connected samples
        surface_indices = set()
        for sample_index in self.region:
            surface_indices = surface_indices | set(sample_points[sample_index].reduced_adjacent_surface_indices)
        self.reduced_adjacent_surface_indices = list(surface_indices)


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
        normal, d, _ = plane
        dist = abs(np.sum(normal * self.position) + d) / np.linalg.norm(normal)
        return dist

    def pixel_polygon_dist(self, polygon):
        triangles = polygon[2]
        min_dist = np.inf
        for triangle in triangles:
            clostest_point = triangle.clostest_point_to(self.position)
            if clostest_point is None:
                # clostest point inside triangle -> point-plane distance is enough
                dist = self.pixel_plane_dist(polygon)
            else:
                dist = np.linalg.norm(self.position - clostest_point)

            if dist < min_dist:
                min_dist = dist
        return min_dist

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
            for surface_index in self.closest_sample.reduced_adjacent_surface_indices:
                dist = self.pixel_polygon_dist(surfaces[surface_index])

                if dist < reduced_min_dist:
                    reduced_min_dist = dist

            self.reduced_dist_surface = reduced_min_dist
        return

    def compute_gray_value(self, rand_num):
        # self.gray_value = 1 - np.power(self.dist_surface, 1/10) + rand_num
        self.reduced_gray_value = -np.power(self.reduced_dist_surface, 1/10) + rand_num
        return


def voronoi_diagram(num=100):
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


def construct_geometry(image_width, num_samples=20, num_removed_surfaces=10):

    border = np.zeros((image_width, image_width, image_width), dtype=int)

    vor, sample_points = voronoi_diagram(num_samples)

    surfaces = {}

    reduced_ridge_point_indices = np.random.choice(range(len(vor.ridge_points)), num_removed_surfaces)

    for surface_index, surface_vertex_indices in enumerate(vor.ridge_vertices):

        if not surface_vertex_indices:
            continue
        if -1 not in surface_vertex_indices:

            # add all the triangles in the given surface
            triangles = []
            num_verts = len(surface_vertex_indices)

            if num_verts == 3:
                # triangle
                verts = vor.vertices[surface_vertex_indices]
                triangles.append(Triangle(verts[0], verts[1], verts[2]))

                if surface_index not in reduced_ridge_point_indices:
                    sampels = TriangleSamples(verts[0], verts[1], verts[2])
                    coords = sampels.get_triangle_samples(image_width)
                    border[coords[:, 0], coords[:, 1], coords[:, 2]] = 1

            elif num_verts > 3:
                # convex polygon
                fixed_vert = vor.vertices[surface_vertex_indices[0]]
                for itr in range(2, num_verts):
                    verts = np.array([
                        fixed_vert,
                        vor.vertices[surface_vertex_indices[itr - 1]],
                        vor.vertices[surface_vertex_indices[itr]]
                    ])
                    triangles.append(Triangle(verts[0], verts[1], verts[2]))

                    if surface_index not in reduced_ridge_point_indices:
                        sampels = TriangleSamples(verts[0], verts[1], verts[2])
                        coords = sampels.get_triangle_samples(image_width)
                        border[coords[:, 0], coords[:, 1], coords[:, 2]] = surface_index

            # compute normal-distance pair
            """
            Adapted from https://stackoverflow.com/questions/53698635/how-to-define-a-plane-with-3-points-and-plot-it-in-3d
            (Reblochon Masque)

            """
            x0, y0, z0 = vor.vertices[surface_vertex_indices[0]]
            x1, y1, z1 = vor.vertices[surface_vertex_indices[1]]
            x2, y2, z2 = vor.vertices[surface_vertex_indices[2]]

            ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]
            vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]

            u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]

            point = np.array(vor.vertices[surface_vertex_indices[0]])
            normal = np.array(u_cross_v)

            d = -point.dot(normal)
            surfaces[surface_index] = (normal, d, triangles)


            # add adjacent surfaces to each sample point
            split_points = vor.ridge_points[surface_index]

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
        # TODO: only process one sample of the region and share information with other samples
        # directly_connected_samples have to be built before for all samples
        sample_point.grow_connected_region(sample_points)
        sample_point.compare_surfaces()

    return vor, sample_points, surfaces, border


def create_false_joins_image_sampling(image_width=50, num_samples=20, num_removed_surfaces=20, sigmas=[0.02], verbose=False):
    start_time = time.time()

    # construct images
    ground_truth = -np.ones((image_width+1, image_width+1, image_width+1), dtype=int)

    # create Voronoi diagram
    vor, sample_points, surfaces, border = construct_geometry(image_width, num_samples, num_removed_surfaces)

    # compute distances to nearest curves
    distance = ndimage.distance_transform_edt(border == 0) / image_width
    distance = np.power(distance, 1/5)

    images = []
    for sigma in sigmas:
        noise_generator = np.random.normal(0, sigma, size=(image_width, image_width, image_width))

        # add noise
        image = distance + noise_generator

        # reformat image to 8bit
        image = np.interp(image, (image.min(), image.max()), (0, 255)).astype(np.uint8)
        image = 255 - image

        images.append(image)

    coords = np.vstack([np.around((sample.position * image_width)).astype(int) for sample in sample_points[:num_samples]])
    for i in range(num_samples):
        ground_truth[coords[i, 0], coords[i, 1], coords[i, 2]] = min(sample_points[i].region)

    _, indices = ndimage.distance_transform_edt(ground_truth == -1, return_indices=True)

    ground_truth = ground_truth[indices[0], indices[1], indices[2]]
    ground_truth = ground_truth[:-1, :-1, :-1]

    if verbose:
        print("--- %s seconds ---" % (time.time() - start_time))

        fig, ax = plt.subplots(2, 1)
        tracker = IndexTracker(ax, image_minnoise=images[0], image_maxnoise=images[-1])
        fig.suptitle('Use scroll wheel to navigate slices \nImage dimensions: ({}, {}, {}) \n'
                     .format(image_width, image_width, image_width))
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()

    return images, ground_truth


def create_false_joins_image_iterating(image_width=50, num_samples=20, num_removed_surfaces=20, sigma=0.02, verbose=False):
    start_time = time.time()

    # create Voronoi diagram
    vor, sample_points, surfaces, border = construct_geometry(image_width, num_samples, num_removed_surfaces)

    # construct images
    ground_truth_reduced = np.zeros((image_width, image_width, image_width), dtype=int)
    noise_reduced = np.zeros(ground_truth_reduced.shape)
    noise_generator = np.random.normal(0, sigma, size=ground_truth_reduced.shape)
    for i in range(image_width):
        print('Progress: Layer', i)
        for j in range(image_width):
            for k in range(image_width):
                pixel = Pixel(np.array([i, j, k]) / image_width)
                pixel.min_dist_sample(sample_points=sample_points)

                ground_truth_reduced[i, j, k] = min(pixel.closest_sample.region)

                pixel.min_dist_surface(surfaces=surfaces)
                pixel.compute_gray_value(rand_num=noise_generator[i, j, k])
                noise_reduced[i, j, k] = pixel.reduced_gray_value

    # reformat image to 8bit
    noise_reduced = np.interp(noise_reduced, (noise_reduced.min(), noise_reduced.max()), (0, 255)).astype(np.uint8)

    if verbose:
        print("--- %s seconds ---" % (time.time() - start_time))

        fig, ax = plt.subplots(2, 1)
        tracker = IndexTracker(ax, true=ground_truth_reduced, img=noise_reduced)
        fig.suptitle('Use scroll wheel to navigate slices \nImage dimensions: ({}, {}, {}) \n'
                     .format(image_width, image_width, image_width))
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()

    return noise_reduced, ground_truth_reduced


def main():
    image_width = 300

    # number of samples in the Voronoi diagram
    num_samples = 1500  # 1500

    # number of surfaces that are removed between random cells
    num_removed_surfaces = 1000  # 1000

    sigmas = [0.02, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    verbose = False

    sampling = True
    to_dataset = True
    size_dataset = 10

    if to_dataset:
        times = []
        for i in range(size_dataset):
            start_time = time.time()
            try:
                os.mkdir('src/storage/datasets/false_joins/Sample' + str(i))
            except OSError:
                print("Already existent")
            else:
                print("Successfully created.")

            if sampling:
                images, ground_truth = create_false_joins_image_sampling(
                    image_width=image_width,
                    num_samples=num_samples,
                    num_removed_surfaces=num_removed_surfaces,
                    sigmas=sigmas,
                    verbose=verbose
                )
                for j, sigma in enumerate(sigmas):
                    np.save('src/storage/datasets/false_joins/Sample' + str(i) + '/image_sigma' + str(sigma),
                            images[j])
            else:
                image, ground_truth = create_false_joins_image_iterating(
                    image_width=image_width,
                    num_samples=num_samples,
                    num_removed_surfaces=num_removed_surfaces,
                    sigma=sigmas[0],
                    verbose=verbose
                )
                np.save('src/storage/datasets/false_joins/Sample' + str(i) + '/image_sigma' + str(sigmas[0]), image)

            np.save('src/storage/datasets/false_joins/Sample' + str(i) + '/ground_truth', ground_truth)

            times.append(time.time() - start_time)
        print("Best Time: ", np.min(times))
        print("Worst Time", np.max(times))
        print("Mean Time: ", np.mean(times))
    else:
        if sampling:
            images, ground_truth = create_false_joins_image_sampling(
                image_width=image_width,
                num_samples=num_samples,
                num_removed_surfaces=num_removed_surfaces,
                sigmas=sigmas,
                verbose=verbose
            )
            for j, sigma in enumerate(sigmas):
                np.save('src/storage/false_joins/image_sigma' + str(sigma),
                        images[j])
        else:
            image, ground_truth = create_false_joins_image_iterating(
                image_width=image_width,
                num_samples=num_samples,
                num_removed_surfaces=num_removed_surfaces,
                sigma=sigmas[0],
                verbose=verbose
            )
            np.save('src/storage/false_joins/image', image)

        np.save('src/storage/false_joins/ground_truth', ground_truth)
