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
        self.num_nodes = len(graph.nodes)
        self.num_classes = num_classes

        self.graph = graph
        edge_vec_length = len(self.graph.edges) * self.num_classes ** 2

        self.node_costs = np.ones((self.num_nodes, self.num_classes))
        self.node_costs[:, 1] = 0
        self.cut_costs = np.zeros(edge_vec_length)
        self.join_costs = np.zeros(edge_vec_length)

    def get_node_cost(self, node, cls):
        return self.node_costs[node, cls]

    def get_cut_cost(self, node0, node1, cls0, cls1):
        if node1 < node0:
            node0, node1 = node1, node0
        index = self.edge_cost_index(
            edge_index=self.graph.find_edge(node0, node1),
            cls0=cls0,
            cls1=cls1
        )
        return self.cut_costs[index]

    def get_join_cost(self, node0, node1, cls0, cls1):
        if node1 < node0:
            node0, node1 = node1, node0
        index = self.edge_cost_index(
            edge_index=self.graph.find_edge(node0, node1),
            cls0=cls0,
            cls1=cls1
        )
        return self.join_costs[index]

    def edge_cost_index(self, edge_index, cls0, cls1):
        return edge_index * self.num_classes**2 + cls0 * self.num_classes + cls1

    def compute_objective_value(self, solution):
        objective_value = 0
        for node_itr in range(self.num_nodes):
            #print("Node", node_itr)
            objective_value += self.get_node_cost(node_itr, solution.solution[node_itr].class_index)

        for edge_itr in range(len(self.graph.edges)):
            #print("Edge", edge_itr)
            node0 = self.graph.edges[edge_itr].node0
            node1 = self.graph.edges[edge_itr].node1

            cls0 = solution.solution[node0].class_index
            cls1 = solution.solution[node1].class_index

            if solution.solution[node0].component_index == solution.solution[node1].component_index:
                objective_value += self.get_join_cost(node0, node1, cls0, cls1)
            else:
                objective_value += self.get_cut_cost(node0, node1, cls0, cls1)
        return objective_value