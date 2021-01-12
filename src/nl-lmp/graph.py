# License
# Copyright © by Bjoern Andres (bjoern@andres.sc).
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# The name of the author must not be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


class Node:
    def __init__(self, id):
        self.id = id
        self.incoming_edge_indices = []
        self.outgoing_edge_indices = []


class Edge:
    def __init__(self, node0, node1):
        self.node0 = node0
        self.node1 = node1

    def __repr__(self):
        return "({}, {})".format(self.node0, self.node1)


class Graph:
    def __init__(self, n):
        self.nodes = []
        for i in range(n):
            self.nodes.append(Node(i))
        self.edges = []

    def __len__(self):
        return len(self.nodes)

    def add_node(self):
        self.nodes.append(Node(len(self.nodes)))

    def add_edge(self, node0, node1):
        assert node0 < len(self.nodes), node1 < len(self.nodes)
        edge = Edge(node0, node1)
        self.edges.append(edge)
        # push edge position in edges list to incident nodes
        self.nodes[node0].outgoing_edge_indices.append(len(self.edges) - 1)
        self.nodes[node1].incoming_edge_indices.append(len(self.edges) - 1)

    def find_edge(self, node0, node1):
        assert node0 < len(self.nodes)
        assert node1 < len(self.nodes)
        # find index of directed edge between the two vertices
        for outgoing_edge_index in self.nodes[node0].outgoing_edge_indices:
            if self.edges[outgoing_edge_index].node1 == node1:
                return outgoing_edge_index
        print("No edge found")
        return
