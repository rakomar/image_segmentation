# License
# Copyright © by Bjoern Andres (bjoern@andres.sc).
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# The name of the author must not be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def update_labeling(problem, solution):
    # class LabelSwap:  # not needed for alternating k-l
    #     def __init__(self, node, new_label, gain):
    #         self.node = node
    #         self.new_label = new_label
    #         self.gain = gain

    graph = problem.graph
    gains = []

    # best gain needs to be negative for an update since it is a minimization problem
    best_gain = 0
    best_label = 0
    best_node = -1

    for node_obj in graph.nodes:

        node = node_obj.id
        original_label = solution.solution[node].class_index

        # node gains
        for label in range(problem.num_classes):
            gain = problem.get_node_cost(node, label) - problem.get_node_cost(node, original_label)

            # incoming edge gains
            for edge in node_obj.incoming_edge_indices:
                neighbor_node = graph.edges[edge].node0
                neighbor_label = solution.solution[neighbor_node].class_index

                if label == neighbor_label:
                    gain += problem.get_join_cost(neighbor_node, node, neighbor_label, label) - \
                            problem.get_join_cost(neighbor_node, node, neighbor_label, original_label)
                else:
                    gain += problem.get_cut_cost(neighbor_node, node, neighbor_label, label) - \
                            problem.get_cut_cost(neighbor_node, node, neighbor_label, original_label)

            # outgoing edge gains
            for edge in node_obj.outgoing_edge_indices:
                neighbor_node = graph.edges[edge].node1
                neighbor_label = solution.solution[neighbor_node].class_index

                if label == neighbor_label:
                    gain += problem.get_join_cost(node, neighbor_node, original_label, neighbor_label) - \
                            problem.get_join_cost(node, neighbor_node, label, neighbor_label)
                else:
                    gain += problem.get_cut_cost(node, neighbor_node, original_label, neighbor_label) - \
                            problem.get_cut_cost(node, neighbor_node, label, neighbor_label)

            if gain < best_gain:
                best_gain = gain
                best_label = label
                best_node = node

            gains.append(gain)

    if best_node != -1:
        solution.solution[best_node].class_index = best_label
    return best_gain
