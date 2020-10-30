import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scipy.spatial import Voronoi
import cv2

def voronoi_centers(num=1000):
    # sample a number of points in the unit cube
    samples = np.random.rand(num, 3)
    return samples


def _adjust_bounds(ax, points):
    # as in scipy.spatial._plotutils._adjust_bounds
    margin = 0.1 * points.ptp(axis=0)
    xy_min = points.min(axis=0) - margin
    xy_max = points.max(axis=0) + margin
    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])


def draw_collection_lines(verts):
    # draw the edge lines for a polygon given by the vertices
    print(type(verts))
    for itr in range(0, len(verts)):
        plt.plot(
            [verts[itr-1, 0], verts[itr, 0]],
            [verts[itr-1, 1], verts[itr, 1]],
            [verts[itr-1, 2], verts[itr, 2]],
            color="k"
        )

centers = voronoi_centers(8)
vor = Voronoi(centers)

fig = plt.figure()
ax = plt.subplot(111, projection="3d")

# draw scatter plots for sample points (Voronoi centers) and Voronoi vertices
ax.scatter(vor.points[:, 0], vor.points[:, 1], vor.points[:, 2], c='#eb7434', marker="o")
ax.scatter(vor.vertices[:, 0], vor.vertices[:, 1], vor.vertices[:, 2], c="#393799", marker="^")


# draw Voronoi vertex triple surfaces
for surface_indices in vor.ridge_vertices:

    if -1 not in surface_indices:
        num_verts = len(surface_indices);

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


# points = np.random.rand(10,2) #random
# from scipy.spatial import Voronoi, voronoi_plot_2d
# vor = Voronoi(points)
#
# fig = voronoi_plot_2d(vor)
#
# fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',
#                 line_width=2, line_alpha=0.6, point_size=2)
# plt.show()
#
# cv2.imwrite("./test.jpg", fig)