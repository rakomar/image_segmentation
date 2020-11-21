import numpy as np
import bezier
import matplotlib.pyplot as plt
import time


class BezierCurve:
    def __init__(self, bezier_module_curve):
        self.bezier_module_curve = bezier_module_curve

        # compute equidistant sample points along curve
        t_regular = np.linspace(0, 1, num_curve_samples)
        self.sample_points = bezier_module_curve.evaluate_multi(t_regular).transpose()

        self.min_curve_dist = None  # not up to date after more curves are added

    def curve_curve_dists(self, curves):
        """
        Compute minimal distance of the current curve to all other curves.

        """
        min_dist = np.inf
        for own_sample in self.sample_points:
            for curve in curves:  # current curve itself is not yet in the curves list
                for other_sample in curve.sample_points:
                    dist = np.linalg.norm(own_sample - other_sample)
                    if dist < min_dist:
                        min_dist = dist
        self.min_curve_dist = min_dist


class Pixel:
    def __init__(self, position):
        self.position = position

        self.dist_curve = None

        self.gray_value = None

    def __repr__(self):
        return 'Pixel: ' + str(self.position)

    def min_dist_curve(self, curves):
        min_dist = np.inf
        for curve in curves:
            for sample in curve.sample_points:
                dist = np.linalg.norm(self.position - sample)
                if dist < min_dist:
                    min_dist = dist
        self.dist_curve = min_dist

    def compute_gray_value(self, rand_num):
        self.gray_value = 1 - np.power(self.dist_curve, 1/10) + rand_num
        return


class IndexTracker(object):
    """
    Class to visualize the 3D images.
    Adapted from https://matplotlib.org/gallery/animation/image_slices_viewer.html

    """

    def __init__(self, ax, A, B):
        # A, B, C, D 3D images of shape image_width^3 stored in np.array
        self.ax = ax

        self.A = A
        self.B = B

        rows, cols, self.slices = A.shape
        self.ind = self.slices//2

        self.im0 = ax[0].imshow(self.A[:, :, self.ind])
        self.ax[0].set_title('Original')
        self.im1 = ax[1].imshow(self.B[:, :, self.ind], cmap='gray')
        self.ax[1].set_title('Noise')

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
        self.ax[0].set_ylabel('slice %s' % self.ind)
        self.im1.set_data(self.B[:, :, self.ind])

        self.im0.axes.figure.canvas.draw()


start_time = time.time()

image_width = 64

# minimal distance between two valid curves
d = 0.05

# how many curves should be created and tested for curve-curve distance
num_spline_tries = 30

# how many sample points on each curve to approximate curve-curve distances
num_curve_samples = 100

fig = plt.figure()
ax = fig.gca(projection='3d')

curves = []

for _ in range(num_spline_tries):
    num_knots = np.random.randint(low=4, high=8)
    knots = np.random.uniform(low=-0.5, high=1.5, size=(3, num_knots))

    # https://stackoverflow.com/questions/51803054/
    # how-to-find-points-along-a-3d-spline-curve-in-scipy (xdze2)
    # create Bezier curve from bezier module
    bc = bezier.Curve(nodes=knots, degree=num_knots - 1)

    # create custom BezierCurve and test for distance to other curves
    curve = BezierCurve(bc)
    curve.curve_curve_dists(curves)

    if curve.min_curve_dist < d:
        del curve

    else:
        curves.append(curve)

        #ax.plot(*knots, '-o')
        ax.plot(curve.sample_points[:, 0], curve.sample_points[:, 1], curve.sample_points[:, 2])
        ax.plot(curve.sample_points[:, 0], curve.sample_points[:, 1], curve.sample_points[:, 2], 'ok', ms=0.8,)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

plt.show()

image = np.zeros((image_width, image_width, image_width))
noise_generator = np.random.normal(0, 0.02, size=image.shape) * 1

pixels = []  # list of length image_width^3

for i in range(image_width):
    print('Progress: Layer', i)
    for j in range(image_width):
        for k in range(image_width):
            pixel = Pixel(np.array([i, j, k]) / image_width)
            pixels.append(pixel)

            pixel.min_dist_curve(curves=curves)
            pixel.compute_gray_value(rand_num=noise_generator[i, j, k])

            image[i, j, k] = pixel.gray_value


image = np.interp(image, (image.min(), image.max()), (0, 255)).astype(np.uint8)

print("--- %s seconds ---" % (time.time() - start_time))

fig, ax = plt.subplots(2, 1)
tracker = IndexTracker(ax, image, image)
fig.suptitle('Use scroll wheel to navigate slices \nImage dimensions: ({}, {}, {})'
             .format(image_width, image_width, image_width))
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()
