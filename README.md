# Extreme Image Segmentation

This repository contains the code and report for the Research Project on the topic of "Extreme Image Segmentation" at the Chair of Machine Learning for Computer Vision of the Faculty of Computer Science at the university TU Dresden.

It is concerned with the influence of False Joins and False Cuts in image segmentation. A False Join refers to the incorrect joining of two pixels that in reality belong to different components and a False Cuts refers to the incorrect cutting apart of two pixels that in reality belong to the same object. The effects of these problem types are analyzed on the example of simple watershed segmentation. 

To facilitate the analysis, the code provides ways to artificially sample 3D images containing examples that highly emphasize the two error classes. The images are created with the help of a Voronoi diagram (False Joins) and Bézier Curves (False Cuts).

## Content
The code contains utilities for the creation of the error class 3D images and segmentation algorithms to segment those images.
- Research Project Report
- tasks: artificially sample 3D images
- watershed: watershed algorithm to perform image segmentation 
- nl-lmp: contains a python implementation of solving algorithms for the NL-LMP problem, adapted from [Graphs and Graph Algorithms in C++](https://github.com/bjoern-andres/graph). The NL-LMP problem was proposed in

```
@inproceedings{levinkov2017joint,
  title={Joint graph decomposition \& node labeling: Problem, algorithms, applications},
  author={Levinkov, Evgeny and Uhrig, Jonas and Tang, Siyu and Omran, Mohamed and Insafutdinov, Eldar and Kirillov, Alexander and Rother, Carsten and Brox, Thomas and Schiele, Bernt and Andres, Bjoern},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6012--6020},
  year={2017}
}
```
