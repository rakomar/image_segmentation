# License
# Copyright © by Bjoern Andres (bjoern@andres.sc).
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# The name of the author must not be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Not updated with current data pieline.

"""


import numpy as np
import time
from sklearn.metrics import rand_score, mutual_info_score
from scipy.stats import entropy

from graph import Graph, CompleteGraph
from problem import Problem
from solution import Solution
from update_labeling import update_labeling
from update_multicut import update_multicut


image = np.load("../storage/datasets/false_joins/Sample0/image_sigma0.02.npy")
ground_truth = np.load("../storage/datasets/false_joins/Sample0/ground_truth.npy")
image_width = image.shape[0]

graph = Graph(image_width ** 3)

# # direct adjacency graph
# for k in range(image_width):
#     for j in range(image_width):
#         for i in range(image_width):
#             if i > 0:
#                 graph.add_edge((i - 1) * image_width ** 2 + j * image_width + k,
#                                i * image_width ** 2 + j * image_width + k)
#
#             if j > 0:
#                 graph.add_edge(i * image_width ** 2 + (j - 1) * image_width + k,
#                                i * image_width ** 2 + j * image_width + k)
#             if k > 0:
#                 graph.add_edge(i * image_width ** 2 + j * image_width + (k - 1),
#                                i * image_width ** 2 + j * image_width + k)


# define shift radius
r = 3

# r adjacency graph
for k in range(image_width):
    for j in range(image_width):
        for i in range(image_width):
            for x2 in range(-r, r + 1):  # only iterate the new halfspace
                for x1 in range(-r, r + 1):
                    for x0 in range(-r, r + 1):
                        if (x0, x1, x2) != (0, 0, 0) and 0 <= (i + x0) < image_width and 0 <= (j + x1) < image_width and 0 <= (k + x2) < image_width:
                            graph.add_edge(
                                i + j * image_width + k * image_width ** 2,
                                (i + x0) + (j + x1) * image_width + (k + x2) * image_width ** 2
                           )

# # complete graph
# for i in range(image_width ** 3):
#     for j in range(i + 1, image_width ** 3):
#         graph.add_edge(i, j)

# for i in range(len(graph.edges)):
#     print(graph.edges[i])


print(graph.num_edges)

num_classes = 1

true_solution = Solution(array=ground_truth)

problem = Problem(graph=graph, num_classes=num_classes)
problem.compute_edge_costs(image=image, ground_truth=ground_truth)

pred_solution = Solution(len(graph.nodes))

print(pred_solution)

for i in range(1):

    # update_labeling(problem=p, solution=s)
    #
    # x = p.compute_objective_value(s)
    # print("best gain", x)
    # print(s)

    start_time = time.time()

    val = update_multicut(problem=problem, solution=pred_solution)

    print("--- %s seconds ---" % (time.time() - start_time))

    print("num_parts: ", len(set([c.component_index for c in pred_solution.solution])))
    print(pred_solution)

    pred_array = pred_solution.to_numpy_array()

    # compute rand index
    true_labels = [elem.component_index for elem in true_solution.solution]
    pred_labels = [elem.component_index for elem in pred_solution.solution]

    rand_index = rand_score(true_labels, pred_labels)

    # compute variation of information
    mutual_information = mutual_info_score(true_labels, pred_labels)

    _, counts = np.unique(true_labels, return_counts=True)
    true_entropy = entropy(counts)
    _, counts = np.unique(pred_labels, return_counts=True)
    pred_entropy = entropy(counts)

    variation_of_information = true_entropy + pred_entropy - 2 * mutual_information

    print("Rand_index: ", rand_index)
    print("Variation of Information", variation_of_information)