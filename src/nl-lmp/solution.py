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


class Solution:

    class Element:
        def __init__(self, class_index, component_index):
            self.class_index = class_index
            self.component_index = component_index

        def __repr__(self):
            #return "(Cls: " + str(self.class_index) + ", Comp: " + str(self.component_index) + ")"
            return "P: " + str(self.component_index)

    def __init__(self, n=None, array=None):
        self.solution = []

        # create newly initialized solution
        if n:
            for i in range(n):
                self.solution.append(self.Element(0, 0))

        # construct solution from (image) array
        elif array is not None:
            width = array.shape[0]
            for k in range(width):
                for j in range(width):
                    for i in range(width):
                        self.solution.append(self.Element(0, array[i, j, k]))

        else:
            print("Provide initialization information.")

    def __len__(self):
        return len(self.solution)

    def __repr__(self):
        return "Solution: " + str([self.solution[i] for i in range(len(self.solution))])

    def to_numpy_array(self):
        width = int(round(len(self.solution) ** (1/3)))
        assert width ** 3 == len(self.solution)

        arr = np.zeros((width, width, width), dtype=int)
        for i in range(width):
            for j in range(width):
                for k in range(width):
                    arr[i, j, k] = self.solution[i + j * width + k * width**2].component_index
        return arr
