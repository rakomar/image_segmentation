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

# settings
num_outer_iterations = 1
num_inner_iterations = 1
epsilon = 0.00001


def update_multicut(problem, solution):
    objective_value = problem.compute_objective_value(solution)

    component_labels = []
    for node in range(problem.num_nodes):
        component_labels.append(solution.solution[node].component_index)

    edge_costs = []
    for edge in problem.graph.edges:
        node0 = edge.node0
        cls0 = solution.solution[node0].class_index
        node1 = edge.node1
        cls1 = solution.solution[node1].class_index
        val = problem.get_cut_cost(node0, node1, cls0, cls1) - problem.get_join_cost(node0, node1, cls0, cls1)
        edge_costs.append(val)

    new_component_labels = kernighan_lin(problem.graph, edge_costs, component_labels)

    for i in range(len(solution)):
        solution.solution[i].component_index = new_component_labels[i]

    new_objective_value = problem.compute_objective_value(solution)

    return new_objective_value - objective_value


def kernighan_lin(graph, edge_costs, component_labels):

    class TwoCutBuffer:
        def __init__(self, graph):
            size_graph = len(graph)
            self.differences = [0] * size_graph
            self.is_moved = [None] * size_graph
            self.referenced_by = [0] * size_graph
            self.vertex_labels = [None] * size_graph

            self.border = None
            self.max_not_used_label = None

    buffer = TwoCutBuffer(graph)

    def update_bipartition(component_A, component_B, graph, buffer, edge_costs):

        class Move:
            def __init__(self):
                self.vertex = -1
                self.difference = -np.inf
                self.new_label = None

        gain_from_merging = 0

        def compute_difference(component_A, label_A, label_B, graph, buffer, edge_costs, gain_from_merging):
            for i in range(len(component_A)):
                difference_interior = 0
                difference_exterior = 0
                reference_counter = 0

                node_obj = graph.nodes[component_A[i]]

                for edge in node_obj.incoming_edge_indices:
                    neighbor_node = graph.edges[edge].node0
                    neighbor_label = buffer.vertex_labels[neighbor_node]

                    if neighbor_label == label_A:
                        difference_interior += edge_costs[edge]
                    elif neighbor_label == label_B:
                        difference_exterior += edge_costs[edge]
                        reference_counter += 1

                for edge in node_obj.outgoing_edge_indices:
                    neighbor_node = graph.edges[edge].node1
                    neighbor_label = buffer.vertex_labels[neighbor_node]

                    if neighbor_label == label_A:
                        difference_interior += edge_costs[edge]
                    elif neighbor_label == label_B:
                        difference_exterior += edge_costs[edge]
                        reference_counter += 1

                buffer.differences[component_A[i]] = difference_exterior - difference_interior
                buffer.referenced_by[component_A[i]] = reference_counter
                buffer.is_moved[component_A[i]] = 0

                gain_from_merging += difference_exterior
            return

        if not component_A:
            return 0

        label_A = buffer.vertex_labels[component_A[0]]
        if component_B:
            label_B = buffer.vertex_labels[component_B[0]]
        else:
            label_B = buffer.max_not_used_label

        compute_difference(component_A, label_A, label_B, graph, buffer, edge_costs, gain_from_merging)
        compute_difference(component_B, label_B, label_A, graph, buffer, edge_costs, gain_from_merging)
        gain_from_merging /= 2

        buffer.border = []

        for node in component_A:
            if buffer.referenced_by[node] > 0:
                buffer.border.append(node)

        for node in component_B:
            if buffer.referenced_by[node] > 0:
                buffer.border.append(node)

        moves = []
        cumulative_difference = 0
        max_move = (-np.inf, 0)

        for iteration in range(num_inner_iterations):
            m = Move()

            if not component_B and iteration == 0:
                for node in component_A:
                    if buffer.differences[node] > m.difference:
                        m.difference = buffer.differences[node]
                        m.vertex = node
            else:
                size = len(buffer.border)

                i = 0
                while i < size:
                    if buffer.referenced_by[buffer.border[i]] == 0:
                        size -= 1
                        buffer.border[i], buffer.border[size] = buffer.border[size], buffer.border[i]
                    else:
                        if buffer.differences[buffer.border[i]] > m.difference:
                            m.difference = buffer.differences[buffer.border[i]]
                            m.vertex = buffer.border[i]
                        i += 1

                del buffer.border[size:]  # remove all elements after size-th element

            if m.vertex == -1:
                break

            old_label = buffer.vertex_labels[m.vertex]
            if old_label == label_A:
                m.new_label = label_B
            else:
                m.new_label = label_A

            # update differences and references
            for edge in graph.nodes[m.vertex].incoming_edge_indices:
                neighbor_node = graph.edges[edge].node0

                if buffer.is_moved[neighbor_node]:
                    continue

                neighbor_label = buffer.vertex_labels[neighbor_node]
                # edge to an element of the new set
                if neighbor_label == m.new_label:
                    buffer.differences[neighbor_node] -= 2 * edge_costs[edge]
                    buffer.referenced_by[neighbor_node] -= 1

                # edge to an element of the old set
                elif neighbor_label == old_label:
                    buffer.differences[neighbor_node] += 2 * edge_costs[edge]
                    buffer.referenced_by[neighbor_node] += 1

                    if buffer.referenced_by[neighbor_node] == 1:
                        buffer.border.append(neighbor_node)

            for edge in graph.nodes[m.vertex].outgoing_edge_indices:
                neighbor_node = graph.edges[edge].node1

                if buffer.is_moved[neighbor_node]:
                    continue

                neighbor_label = buffer.vertex_labels[neighbor_node]
                # edge to an element of the new set
                if neighbor_label == m.new_label:
                    buffer.differences[neighbor_node] -= 2 * edge_costs[edge]
                    buffer.referenced_by[neighbor_node] -= 1

                # edge to an element of the old set
                elif neighbor_label == old_label:
                    buffer.differences[neighbor_node] += 2 * edge_costs[edge]
                    buffer.referenced_by[neighbor_node] += 1

                    if buffer.referenced_by[neighbor_node] == 1:
                        buffer.border.append(neighbor_node)

            buffer.vertex_labels[m.vertex] = m.new_label
            buffer.referenced_by[m.vertex] = 0
            buffer.differences[m.vertex] = -np.inf
            buffer.is_moved[m.vertex] = 1
            moves.append(m)

            cumulative_difference += m.difference

            if cumulative_difference > max_move[0]:
                max_move = (cumulative_difference, len(moves))

        if gain_from_merging > max_move[0] and gain_from_merging > epsilon:
            component_A.extend(component_B)

            for node in component_A:
                buffer.vertex_labels[node] = label_A
            for node in component_B:
                buffer.vertex_labels[node] = label_A

            component_B = []
            return gain_from_merging, component_A, component_B  # TODO: check if comp A and comp B have to be returned

        elif max_move[0] > epsilon:
            for i in range(max_move[1], len(moves)):
                buffer.is_moved[moves[i].vertex] = 0

                if moves[i].new_label == label_B:
                    buffer.vertex_labels[moves[i].vertex] = label_A
                else:
                    buffer.vertex_labels[moves[i].vertex] = label_B

            if not component_B:
                buffer.max_not_used_label += 1

            component_A = [elem for elem in component_A if not buffer.is_moved[elem]]  # TODO: right behavior?
            component_B = [elem for elem in component_B if not buffer.is_moved[elem]]

            for i in range(max_move[1]):
                # move vertex to other set
                if moves[i].new_label == label_B:
                    component_B.append(moves[i].vertex)
                else:
                    component_A.append(moves[i].vertex)
            return max_move[0], component_A, component_B

        else:
            for i in range(len(moves)):
                if moves[i].new_label == label_B:
                    buffer.vertex_labels[moves[i].vertex] = label_A
                else:
                    buffer.vertex_labels[moves[i].vertex] = label_B

        return 0, component_A, component_B

    # compute initial edge objective value
    starting_objective_value = 0
    for i in range(len(graph.edges)):
        node0 = graph.edges[i].node0
        node1 = graph.edges[i].node1
        if component_labels[node0] != component_labels[node1]:
            starting_objective_value += edge_costs[i]

    num_components = max(component_labels) + 1

    # build partitions
    partitions = [[] for _ in range(num_components)]
    for i in range(len(component_labels)):
        partitions[component_labels[i]].append(i)
        buffer.vertex_labels[i] = component_labels[i]

    buffer.max_not_used_label = len(partitions)

    print("Starting obj val: ", starting_objective_value)

    last_good_vertex_labels = buffer.vertex_labels

    # keep track of visited nodes for BFS/DFS
    visited = [0] * len(graph)

    changed = [1] * num_components

    # iteratively update bipartitions to reduce objective value
    for _ in range(num_outer_iterations):

        objective_value_reduction = 0

        adjacent_components = [set() for _ in range(num_components)]

        for edge in graph.edges:
            label0 = buffer.vertex_labels[edge.node0]
            label1 = buffer.vertex_labels[edge.node1]

            if label0 != label1:
                adjacent_components[min(label0, label1)].add(max(label0, label1))

        for component in range(num_components):
            if partitions[component]:
                for other_component in adjacent_components[component]:
                    if partitions[other_component] and (changed[component] or changed[other_component]):

                        ret, partitions[component], partitions[other_component] = update_bipartition(partitions[component], partitions[other_component], graph, buffer, edge_costs)

                        if ret > epsilon:
                            changed[component] = 1
                            changed[other_component] = 1

                        objective_value_reduction += ret

                        if len(partitions[component]) == 0:
                            break

        ee = objective_value_reduction

        # remove partitions that became empty after the previous step
        partitions = [partition for partition in partitions if partition]

        # try to introduce new partitions
        num_partitions = len(partitions)
        for i in range(num_partitions):
            if not changed[i]:
                continue

            while True:
                new_set = []
                val, partitions[i], new_set = update_bipartition(partitions[i], new_set, graph, buffer, edge_costs)  # TODO: why introduce new partitions all the time
                objective_value_reduction += val

                if not new_set:
                    break

                partitions.append(new_set)

        # if !visitor(buffer.vertex_labels)
        #     break

        if objective_value_reduction == 0:
            break

        stack = []

        partitions = []
        num_components = 0

        # do connected component labeling on the original graph and form new partitions
        for i in range(len(graph)):
            if not visited[i]:
                stack.append(i)
                visited[i] = 1

                label = buffer.vertex_labels[i]
                buffer.referenced_by[i] = num_components

                partitions.append([])
                partitions[-1].append(i)

                while stack:
                    node = stack[-1]
                    del stack[-1]

                    for edge in graph.nodes[node].incoming_edge_indices:
                        neighbor_node = graph.edges[edge].node0

                        if buffer.vertex_labels[neighbor_node] == label and not visited[neighbor_node]:
                            stack.append(neighbor_node)
                            visited[neighbor_node] = 1
                            buffer.referenced_by[neighbor_node] = num_components
                            partitions[-1].append(neighbor_node)

                    for edge in graph.nodes[node].outgoing_edge_indices:
                        neighbor_node = graph.edges[edge].node1

                        if buffer.vertex_labels[neighbor_node] == label and not visited[neighbor_node]:
                            stack.append(neighbor_node)
                            visited[neighbor_node] = 1
                            buffer.referenced_by[neighbor_node] = num_components
                            partitions[-1].append(neighbor_node)

                num_components += 1

        buffer.vertex_labels = buffer.referenced_by
        buffer.max_not_used_label = num_components

        didnt_change = True
        for i in range(len(graph.edges)):
            node0 = graph.edges[i].node0
            node1 = graph.edges[i].node1

            edge_label = (buffer.vertex_labels[node0] != buffer.vertex_labels[node1])

            if edge_label != (last_good_vertex_labels[node0] != last_good_vertex_labels[node1]):
                didnt_change = False

        if didnt_change:
            break

        # check if the shape of some partitions didn't change
        changed = [0] * num_components
        visited = [0] * len(visited)

        for i in range(len(graph)):
            if not visited[i]:
                stack.append(i)
                visited[i] = 1

                label_new = buffer.vertex_labels[i]
                label_old = last_good_vertex_labels[i]

                while stack:
                    node = stack[-1]
                    del stack[-1]

                    for edge in graph.nodes[node].incoming_edge_indices:
                        neighbor_node = edge.node0

                        if last_good_vertex_labels[neighbor_node] == label_old and buffer.vertex_labels[neighbor_node] != label_new:
                            changed[label_new] = 1

                        if visited[neighbor_node]:
                            continue

                        if buffer.vertex_labels[neighbor_node] == label_new:
                            stack.append(neighbor_node)
                            visited[neighbor_node] = 1

                            if last_good_vertex_labels[neighbor_node] != label_old:
                                changed[label_new] = 1

                    for edge in graph.nodes[node].outgoing_edge_indices:
                        neighbor_node = edge.node1

                        if last_good_vertex_labels[neighbor_node] == label_old and buffer.vertex_labels[neighbor_node] != label_new:
                            changed[label_new] = 1

                        if visited[neighbor_node]:
                            continue

                        if buffer.vertex_labels[neighbor_node] == label_new:
                            stack.append(neighbor_node)
                            visited[neighbor_node] = 1

                            if last_good_vertex_labels[neighbor_node] != label_old:
                                changed[label_new] = 1

        last_good_vertex_labels = buffer.vertex_labels

        print("Last good vertex labels: ", last_good_vertex_labels)

    return last_good_vertex_labels


