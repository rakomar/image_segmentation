import numpy as np
import bezier
import matplotlib.pyplot as plt
import time
from math import factorial
import os
from scipy import ndimage

from ..utils import IndexTracker, plot_3d


def newtons_method(f, x0, Df):
    epsilon = 0.001
    max_iter = 25

    out_of_bounds = False
    xi = x0
    for i in range(max_iter):
        # # keep parameter in bounds
        # if xi < 0:
        #     # if out_of_bounds:
        #         #     #     return None
        #         #     # xi = 0.0
        #         #     # out_of_bounds = True
        #         # elif xi > 1:
        #         #     if out_of_bounds:
        #         #         return None
        #         #     xi = 1.0
        #         #     out_of_bounds = True
        #         # else:
        #         #     out_of_bounds = False
        # if xi < 0 or xi > 1:
        #     return None

        fxi = f(xi)
        if abs(fxi) < epsilon:
            # solution with accuracy of epsilon
            return xi

        Dfxi = Df(xi)
        # if Dfxi == 0:
        #     # no solution
        #     return None

        xi = xi - fxi / Dfxi

    # no solution
    return None  # TODO: not found after max_iter iterations


class BezierCurve:
    def __init__(self, bezier_module_curve, num_curve_samples, image_width, extension_rate=0.5):
        self.bezier_module_curve = bezier_module_curve
        n = len(bezier_module_curve.nodes)
        self.num_knots = n

        # compute equidistant sample points along curve for visualizations
        t_regular = np.linspace(0, 1, num_curve_samples)
        self.sample_points = bezier_module_curve.evaluate_multi(t_regular).transpose()

        # round sample points to integer indices and only take geometry within the image boundaries (with extented borders)
        self.rounded_sample_points = np.around(self.sample_points[(self.sample_points.min(axis=1) >= -extension_rate) & (self.sample_points.min(axis=1) <= 1 + extension_rate), :] * image_width).astype(int)
        self.rounded_sample_points = np.unique(self.rounded_sample_points[self.rounded_sample_points.max(axis=1) < int((1 + extension_rate) * image_width)], axis=0) + int(extension_rate * image_width)

        # precomputations for Bezier derivatives
        self.factorials = np.array([factorial(i) for i in range(n)])
        nodes = self.bezier_module_curve.nodes.transpose()
        derivative_q = np.zeros((n - 1, 3))
        second_derivative_q = np.zeros((n - 2, 3))
        for i in range(0, n - 1):
            derivative_q[i] = n * (nodes[i+1] - nodes[i])
            if i < n - 2:
                second_derivative_q[i] = n * (n - 1) * (nodes[i+2] - 2 * nodes[i+1] + nodes[i])
        self.derivative_q = derivative_q
        self.second_derivative_q = second_derivative_q
        self.t = None
        self.b = None

    def curve_curve_dists(self, curves):
        """
        Compute minimal distance of the current curve to all other curves.

        """
        from scipy.spatial.distance import cdist
        min_dist = np.inf
        own_samples = np.array(self.sample_points)
        for curve in curves:
            other_samples = np.array(curve.sample_points)

            dist = np.min(cdist(own_samples, other_samples))
            if dist < min_dist:
                min_dist = dist

        return min_dist

    # def bezier_coefficient(self, n, i, t):
    #     return self.factorials[n] * (t ** i) * ((1 - t) ** (n-i)) / (self.factorials[i] * self.factorials[n - i])

    def compute_b(self, t):
        # if already computed for same t use it again
        if t == self.t:
            return self.b
        # else compute for new t
        else:
            # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/Bezier/bezier-der.html
            n = self.num_knots - 1
            b = np.arange(n)
            b = np.expand_dims(self.factorials[n] * (np.power(t, b)) * (np.power(1 - t, n - b)) / (
                        self.factorials[b] * self.factorials[n - b]), -1)
            self.t = t
            self.b = b
            return b

    def derivative(self, t):
        b = self.compute_b(t)
        return np.sum(self.derivative_q * b, axis=0)

    def second_derivative(self, t):
        b = self.compute_b(t)
        # only use b up to second last entry
        return np.sum(self.second_derivative_q * b[-1:], axis=0)


class Pixel:
    def __init__(self, position):
        self.position = position
        self.dist_curve = None
        self.gray_value = None

    def __repr__(self):
        return 'Pixel: ' + str(self.position)

    def min_dist_curve(self, curves):
        min_dist = np.inf
        t = None
        for index, curve in enumerate(curves):
            f = lambda t: np.sum(np.square(curve.bezier_module_curve.evaluate(t).transpose() - self.position))
            Df = lambda t: 2 * np.sum(curve.derivative(t) * (curve.bezier_module_curve.evaluate(t).transpose() - self.position))
            D2f = lambda t: 2 * np.sum(np.square(curve.derivative(t)) + curve.second_derivative(t) * (curve.bezier_module_curve.evaluate(t).transpose() - self.position))

            # for i in range(curve.num_knots):
            for i in [0, curve.num_knots/2, curve.num_knots]:
                t_best = newtons_method(Df, (i+1) / (curve.num_knots+1), D2f)

                if t_best is not None:
                    dist = f(t_best)
                    if dist < min_dist:
                        min_dist = dist
                        t = t_best

        assert min_dist != np.inf  # TODO: assertion failed
        self.dist_curve = min_dist
        return t

    def compute_gray_value(self, rand_num):
        self.gray_value = 1 - np.power(self.dist_curve, 1/10) + rand_num
        return


def create_false_cuts_image_sampling(image_width, d, num_spline_tries, sigmas=[0.02], extension_rate=0.5, verbose=False):
    start_time = time.time()

    if verbose:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

    width = int((2 * extension_rate + 1) * image_width)

    ground_truth = np.zeros((width, width, width))

    curves = []

    for _ in range(num_spline_tries):
        num_knots = np.random.randint(low=4, high=8)
        knots = np.random.uniform(low=-0.5, high=1.5, size=(3, num_knots))

        # https://stackoverflow.com/questions/51803054/
        # how-to-find-points-along-a-3d-spline-curve-in-scipy (xdze2)
        # create Bezier curve from bezier module
        bc = bezier.Curve(nodes=knots, degree=num_knots - 1)

        # how many sample points on each curve to approximate curve-curve distances
        num_curve_samples = int(bc.length * image_width)

        # create custom BezierCurve and test for distance to other curves
        curve = BezierCurve(bc, num_curve_samples, image_width, extension_rate)
        dist = curve.curve_curve_dists(curves)

        if dist < d:
            del curve

        else:
            curves.append(curve)
            coords = curve.rounded_sample_points
            ground_truth[coords[:, 0], coords[:, 1], coords[:, 2]] = len(curves)  # starting from 1, 0 is background

            if verbose:
                ax.plot(curve.sample_points[:, 0], curve.sample_points[:, 1], curve.sample_points[:, 2])

    # compute distances to nearest curves
    mask = (ground_truth == 0)
    distance, indices = ndimage.distance_transform_edt(mask, return_indices=True)

    # widen the curves in ground truth
    ground_truth = np.where(distance < np.ceil(image_width/100), ground_truth[indices[0], indices[1], indices[2]], 0)

    distance = ndimage.distance_transform_edt(mask) / image_width
    distance = np.power(distance, 1/5)

    width_padding = int(extension_rate * image_width)
    images = []
    for sigma in sigmas:
        noise_generator = np.random.normal(0, sigma, size=ground_truth.shape)

        # add noise
        image = distance + noise_generator

        # resize to actual size
        image = image[width_padding:-width_padding, width_padding:-width_padding, width_padding:-width_padding]
        image = np.interp(image, (image.min(), image.max()), (0, 255)).astype(np.uint8)

        images.append(image)

    ground_truth = ground_truth[width_padding:-width_padding, width_padding:-width_padding,
                   width_padding:-width_padding]

    if verbose:
        print("--- %s seconds ---" % (time.time() - start_time))

        fig, ax = plt.subplots(2, 1)
        tracker = IndexTracker(ax, image_minnoise=images[0], image_maxnoise=images[-1])
        fig.suptitle('Use scroll wheel to navigate slices \nImage dimensions: ({}, {}, {})'
                     .format(image_width, image_width, image_width))
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()

    return images, ground_truth


def create_false_cuts_image_iterating(image_width, d, num_spline_tries, sigma=0.02, extension_rate=None, verbose=False):
    start_time = time.time()

    if verbose:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

    ground_truth = np.zeros((image_width, image_width, image_width))
    noise_generator = np.random.normal(0, sigma, size=ground_truth.shape)
    image = np.zeros((image_width, image_width, image_width))
    # not_found = np.zeros(image.shape)

    curves = []

    for _ in range(num_spline_tries):
        num_knots = np.random.randint(low=4, high=8)
        knots = np.random.uniform(low=-0.5, high=1.5, size=(3, num_knots))

        # https://stackoverflow.com/questions/51803054/
        # how-to-find-points-along-a-3d-spline-curve-in-scipy (xdze2)
        # create Bezier curve from bezier module
        bc = bezier.Curve(nodes=knots, degree=num_knots - 1)

        # how many sample points on each curve to approximate curve-curve distances
        num_curve_samples = int(bc.length * image_width)

        # create custom BezierCurve and test for distance to other curves
        curve = BezierCurve(bc, num_curve_samples, image_width, extension_rate=0)
        dist = curve.curve_curve_dists(curves)

        if dist < d:
            del curve

        else:
            curves.append(curve)
            coords = curve.rounded_sample_points
            ground_truth[coords[:, 0], coords[:, 1], coords[:, 2]] = len(curves)  # starting from 1, 0 is background

            if verbose:
                ax.plot(curve.sample_points[:, 0], curve.sample_points[:, 1], curve.sample_points[:, 2])

    # compute distances to nearest curves
    mask = (ground_truth == 0)
    distance, indices = ndimage.distance_transform_edt(mask, return_indices=True)

    # widen the curves in ground truth
    ground_truth = np.where(distance < np.ceil(image_width/100), ground_truth[indices[0], indices[1], indices[2]], 0)

    for i in range(image_width):
        print('Progress: Layer', i)
        for j in range(image_width):
            for k in range(image_width):
                pixel = Pixel(np.array([i, j, k]) / image_width)

                t = pixel.min_dist_curve(curves=curves)
                # if t == 0.0 or t == 1.0:
                #     not_found[i, j, k] = 1

                pixel.compute_gray_value(rand_num=noise_generator[i, j, k])

                image[i, j, k] = pixel.gray_value

    image = np.interp(image, (image.min(), image.max()), (0, 255)).astype(np.uint8)
    image = 255 - image

    if verbose:
        print("--- %s seconds ---" % (time.time() - start_time))

        fig, ax = plt.subplots(2, 1)
        tracker = IndexTracker(ax, ground_truth=ground_truth, image=image)
        fig.suptitle('Use scroll wheel to navigate slices \nImage dimensions: ({}, {}, {})'
                     .format(image_width, image_width, image_width))
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()

    return image, ground_truth


def main():
    image_width = 300

    # minimal distance between two valid curves
    d = 0.05  # 0.05

    # how many curves should be created and tested for curve-curve distance
    num_spline_tries = 40  # 40

    # size of padding relative to image_width
    extension_rate = 0.1  # 0.1

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
                os.mkdir('src/storage/datasets/false_cuts/Sample' + str(i))
            except OSError:
                print("Already existent")
            else:
                print("Successfully created.")

            if sampling:
                images, ground_truth = create_false_cuts_image_sampling(
                    image_width=image_width,
                    d=d,
                    num_spline_tries=num_spline_tries,
                    sigmas=sigmas,
                    extension_rate=extension_rate,
                    verbose=verbose
                )
                for j, sigma in enumerate(sigmas):
                    np.save('src/storage/datasets/false_cuts/Sample' + str(i) + '/image_sigma' + str(sigma),
                            images[j])
            else:
                image, ground_truth = create_false_cuts_image_iterating(
                    image_width=image_width,
                    d=d,
                    num_spline_tries=num_spline_tries,
                    sigma=sigmas[0],
                    extension_rate=extension_rate,
                    verbose=verbose
                )
                np.save('src/storage/datasets/false_cuts/Sample' + str(i) + '/image_sigma' + str(sigmas[0]), image)

            np.save('src/storage/datasets/false_cuts/Sample' + str(i) + '/ground_truth', ground_truth)

            times.append(time.time() - start_time)
        print("Best Time: ", np.min(times))
        print("Worst Time", np.max(times))
        print("Mean Time: ", np.mean(times))
    else:
        if sampling:
            images, ground_truth = create_false_cuts_image_sampling(
                image_width=image_width,
                d=d,
                num_spline_tries=num_spline_tries,
                sigmas=sigmas,
                extension_rate=extension_rate,
                verbose=verbose
            )
            for j, sigma in enumerate(sigmas):
                np.save('src/storage/false_cuts/image_sigma' + str(sigma),
                        images[j])
        else:
            image, ground_truth = create_false_cuts_image_iterating(
                image_width=image_width,
                d=d,
                num_spline_tries=num_spline_tries,
                sigma=sigmas[0],
                extension_rate=extension_rate,
                verbose=verbose
            )
            np.save('src/storage/false_cuts/image', image)

        np.save('src/storage/false_cuts/ground_truth', ground_truth)
