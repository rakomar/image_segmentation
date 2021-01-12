# License
# Copyright © by Bjoern Andres (bjoern@andres.sc).
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# The name of the author must not be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np


class Problem:

    def __init__(self, graph, num_classes):
        self.num_nodes = len(graph)
        self.num_classes = num_classes

        self.graph = graph
        edge_vec_length = len(self.graph.edges) * self.num_classes ** 2

        self.node_costs = np.zeros((self.num_nodes, self.num_classes))
        self.cut_costs = np.zeros(edge_vec_length)
        self.join_costs = np.zeros(edge_vec_length)

    def get_node_cost(self, node, cls):
        return self.node_costs[node, cls]

    def set_node_cost(self, node, cls, val):
        self.node_costs[node, cls] = val
        return

    def get_cut_cost(self, node0, node1, cls0, cls1):
        if node1 < node0:
            node0, node1 = node1, node0
        index = self.edge_cost_index(
            edge_index=self.graph.find_edge(node0, node1),
            cls0=cls0,
            cls1=cls1
        )
        return self.cut_costs[index]

    def set_cut_cost(self, node0, node1, cls0, cls1, val):
        if node1 < node0:
            node0, node1 = node1, node0
        index = self.edge_cost_index(
            edge_index=self.graph.find_edge(node0, node1),
            cls0=cls0,
            cls1=cls1
        )
        self.cut_costs[index] = val
        return

    def get_join_cost(self, node0, node1, cls0, cls1):
        if node1 < node0:
            node0, node1 = node1, node0
        index = self.edge_cost_index(
            edge_index=self.graph.find_edge(node0, node1),
            cls0=cls0,
            cls1=cls1
        )
        return self.join_costs[index]

    def set_join_cost(self, node0, node1, cls0, cls1, val):
        if node1 < node0:
            node0, node1 = node1, node0
        index = self.edge_cost_index(
            edge_index=self.graph.find_edge(node0, node1),
            cls0=cls0,
            cls1=cls1
        )
        self.join_costs[index] = val
        return

    def edge_cost_index(self, edge_index, cls0, cls1):
        return edge_index * self.num_classes**2 + cls0 * self.num_classes + cls1

    def compute_objective_value(self, solution):
        objective_value = 0
        for node_itr in range(self.num_nodes):
            objective_value += self.get_node_cost(node_itr, solution.solution[node_itr].class_index)

        for edge_itr in range(len(self.graph.edges)):
            node0 = self.graph.edges[edge_itr].node0
            node1 = self.graph.edges[edge_itr].node1

            cls0 = solution.solution[node0].class_index
            cls1 = solution.solution[node1].class_index

            if solution.solution[node0].component_index == solution.solution[node1].component_index:
                objective_value += self.get_join_cost(node0, node1, cls0, cls1)
            else:
                objective_value += self.get_cut_cost(node0, node1, cls0, cls1)
        return objective_value

    def dims_from_index(self, index, width):
        d2 = int(np.floor(index / width ** 2))
        d1 = int(np.floor((index % width ** 2) / width))
        d0 = (index % width ** 2) % width
        return d0, d1, d2

    def compute_edge_costs(self, image, ground_truth):
        arr = np.zeros(image.shape, dtype=int)

        width = image.shape[0]
        count = 0
        for edge in self.graph.edges:
            print(edge)
            # node0 = edge.node0
            # d0, d1, d2 = self.dims_from_index(node0, width)
            # gray_val0 = image[d0, d1, d2]
            #
            # node1 = edge.node1
            # d0, d1, d2 = self.dims_from_index(node1, width)
            # gray_val1 = image[d0, d1, d2]
            #
            # if gray_val0 < 102:
            #     if gray_val1 < 102:
            #         join_cost = 0
            #         cut_cost = 1
            #     else:
            #         join_cost = 1
            #         cut_cost = 0
            # else:
            #     if gray_val1 < 102:
            #         join_cost = 1
            #         cut_cost = 0
            #     else:
            #         join_cost = 0
            #         cut_cost = 1

            node0 = edge.node0
            d0, d1, d2 = self.dims_from_index(node0, width)

            node1 = edge.node1
            d3, d4, d5 = self.dims_from_index(node1, width)

            # join_cost = -10
            # cut_cost = 10

            if ground_truth[d0, d1, d2] == ground_truth[d3, d4, d5]:
                join_cost = 0
                cut_cost = 500
            else:
                count += 1
                join_cost = 1000
                cut_cost = -20
                arr[d0, d1, d2] = 1
                arr[d3, d4, d5] = 1

            # join_cost = 0
            # cut_cost = 20
            # if (d0, d1, d2) == (2, 2, 2) or (d3, d4, d5) == (2, 2, 2):
            #     cut_cost = -20
            # arr[2, 2, 2] = 1
            # if (d2 == 1 and d5 == 2) or (d2 == 2 and d5 == 1):
            #     count += 1
            #     cut_cost = -20
            #     join_cost = 20

            # join_cost = abs(int(gray_val0) - int(gray_val1)) / 255
            # cut_cost = 3*abs(int(gray_val0) - int(gray_val1)) / 255

            self.set_join_cost(
                node0=node0,
                node1=node1,
                cls0=0,
                cls1=0,
                val=join_cost
            )
            self.set_cut_cost(
                node0=node0,
                node1=node1,
                cls0=0,
                cls1=0,
                val=cut_cost
            )
        print(count)
        return arr
