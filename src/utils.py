import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class IndexTracker(object):
    """
    Class to visualize the 3D images.
    Adapted from https://matplotlib.org/gallery/animation/image_slices_viewer.html

    """

    def __init__(self, ax, **kwargs):
        # A, B, C, D 3D images of shape image_width^3 stored in np.array
        assert len(kwargs) > 0
        self.ax = ax
        self.subplots_size = ax.shape

        self.data = []
        for image in kwargs.values():
            self.data.append(image)

        rows, cols, self.slices = self.data[0].shape
        self.ind = self.slices//2

        self.plots = []
        for i, arg in enumerate(kwargs):
            if len(self.subplots_size) == 1:
                self.plots.append(ax[i].imshow(kwargs[arg][:, :, self.ind], cmap="gray"))
                self.ax[i].set_title(arg)
                self.ax[i].set_axis_off()
            elif len(self.subplots_size) == 2:
                self.plots.append(ax[i % 2, int(np.floor(i/2))].imshow(kwargs[arg][:, :, self.ind], cmap='gray'))
                self.ax[i % 2, int(np.floor(i/2))].set_title(arg)

        self.update()

    def onscroll(self, event):
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        for i, plot in enumerate(self.plots):
            plot.set_data(self.data[i][:, :, self.ind])

        if len(self.subplots_size) == 1:
            self.ax[0].set_ylabel('slice %s' % self.ind)
            self.plots[0].axes.figure.canvas.draw()
        elif len(self.subplots_size) == 2:
            self.ax[0, 0].set_ylabel('slice %s' % self.ind)
            self.plots[0].axes.figure.canvas.draw()


def cartesian_product_broadcasted(*arrays):
    """
    http://stackoverflow.com/a/11146645/190597 (senderle)
    """
    import functools
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


def plot_3d(image, mode, pred=None):

    if pred is not None:
        # drawn second plot first
        fig = plt.figure()

        ax = fig.add_subplot(1, 2, 2, projection='3d')

        size = pred.shape[0]
        volume = pred
        x, y, z = cartesian_product_broadcasted(*[np.arange(size, dtype='int16')] * 3).T
        if mode == "joins":
            mask = ((x == 0) | (x == size - 1)
                    | (y == 0) | (y == size - 1)
                    | (z == 0) | (z == size - 1))
        else:
            mask = pred[x, y, z] != 0
        x = x[mask]
        y = y[mask]
        z = z[mask]

        volume = volume.ravel()[mask]

        ax.scatter(x, y, z, c=volume)

        ax = fig.add_subplot(1, 2, 1, projection='3d')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        # ax.set_axis_off()

    size = image.shape[0]
    volume = image
    x, y, z = cartesian_product_broadcasted(*[np.arange(size, dtype='int16')] * 3).T
    if mode == "joins":
        mask = ((x == 0) | (x == size - 1)
                | (y == 0) | (y == size - 1)
                | (z == 0) | (z == size - 1))
    else:
        mask = image[x, y, z] != 0

    x = x[mask]
    y = y[mask]
    z = z[mask]
    volume = volume.ravel()[mask]

    ax.scatter(x, y, z, c=volume, cmap="gnuplot")
    plt.show()
    return fig


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


def plot_3d_graph(vor):
    fig = plt.figure()
    ax = plt.subplot(111, projection="3d")

    # draw scatter plots for sample points (Voronoi centers) and Voronoi vertices
    ax.scatter(vor.points[:, 0], vor.points[:, 1], vor.points[:, 2], c='#eb7434', marker="o")
    ax.scatter(vor.vertices[:, 0], vor.vertices[:, 1], vor.vertices[:, 2], c="#393799", marker="^")

    # draw Voronoi vertex triple surfaces
    for surface_indices in vor.ridge_vertices:

        if -1 not in surface_indices:
            num_verts = len(surface_indices)

            if num_verts == 3:
                # triangle
                verts = vor.vertices[surface_indices]
                triangle_surface = Poly3DCollection(verts=verts, alpha=.25)
                plt.gca().add_collection3d(triangle_surface)

                draw_collection_lines(verts)

            elif num_verts > 3:
                # convex polygon
                fixed_vert = vor.vertices[surface_indices[0]]
                for itr in range(2, num_verts):
                    verts = np.array([
                        fixed_vert,
                        vor.vertices[surface_indices[itr - 1]],
                        vor.vertices[surface_indices[itr]]
                    ])
                    triangle_surface = Poly3DCollection(verts=verts, alpha=.25)
                    plt.gca().add_collection3d(triangle_surface)

                draw_collection_lines(verts)

    ax.set_xlim3d(0.0, 1.0)
    ax.set_ylim3d(0.0, 1.0)
    ax.set_zlim3d(0.0, 1.0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

