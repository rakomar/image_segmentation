import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.morphology import h_maxima, h_minima
from skimage.filters import threshold_otsu
import time
from scipy import ndimage
from sklearn.metrics import rand_score, mutual_info_score, adjusted_rand_score
from scipy.stats import entropy
import os

from ..utils import IndexTracker, plot_3d


def apply_watershed(image, threshold=100, h_value=20, with_mask=True):

    extrema = h_minima(image, h=h_value)
    markers, _ = ndimage.label(extrema)

    if with_mask:
        mask = image < threshold
        result = watershed(image, markers, mask=mask)
    else:
        result = watershed(image, markers)

    return result


def main(mode=None, with_mask=True, only_foreground=False, sigmas=[0.02], h_value=65):

    # mode = "joins"
    # mode = "cuts"

    verbose = False

    from_dataset = True
    size_dataset = 10

    # threshold = threshold_otsu(image)
    threshold = 130

    for sigma in sigmas:

        times = []
        rand_indices = []
        vois = []

        pred_component_nums = []
        gt_component_nums = []

        for i in range(size_dataset):
            start_time = time.time()
            print("Sample ", i)
            if from_dataset:
                image = np.load("src/storage/datasets/false_" + mode + "/Sample" + str(i) + "/image_sigma" + str(sigma) + ".npy")
                ground_truth = np.load("src/storage/datasets/false_" + mode + "/Sample" + str(i) + "/ground_truth.npy")
            else:
                image = np.load("src/storage/false_" + mode + "/image.npy")
                ground_truth = np.load("src/storage/false_" + mode + "/ground_truth.npy")
            image_width = image.shape[0]

            pred = apply_watershed(image, threshold, h_value, with_mask=with_mask)

            pred_component_nums.append(len(np.unique(pred)))
            #print("Num_Pred_Components: ", len(np.unique(pred)))
            gt_component_nums.append(len(np.unique(ground_truth)))
            #print("Num_GT_Components: ", len(np.unique(ground_truth)))

            # compute rand index
            if mode == "cuts" and only_foreground:
                gt_indices = np.nonzero(ground_truth)
                p_indices = np.nonzero(pred)
                shared_indices = [np.concatenate([gt_indices[i], p_indices[i]]) for i in range(len(gt_indices))]
                shared_indices = np.vstack((shared_indices[0], shared_indices[1], shared_indices[2]))
                shared_indices = np.unique(shared_indices, axis=1)

                true_labels = ground_truth[shared_indices[0, :], shared_indices[1, :], shared_indices[2, :]]
                pred_labels = pred[shared_indices[0, :], shared_indices[1, :], shared_indices[2, :]]
            else:
                true_labels = ground_truth.astype(int).flatten()
                pred_labels = pred.astype(int).flatten()

            rand_index = rand_score(true_labels, pred_labels)
            rand_indices.append(rand_index)

            # compute variation of information
            mutual_information = mutual_info_score(true_labels, pred_labels)

            _, counts = np.unique(true_labels, return_counts=True)
            true_entropy = entropy(counts)
            _, counts = np.unique(pred_labels, return_counts=True)
            pred_entropy = entropy(counts)

            variation_of_information = true_entropy + pred_entropy - 2 * mutual_information
            vois.append(variation_of_information)

            if verbose:
                print("Rand_index: ", rand_index)
                print("Variation of Information", variation_of_information)

                fig, ax = plt.subplots(1, 3)
                tracker = IndexTracker(ax, ground_truth=ground_truth, image=image, pred=pred)
                fig.suptitle('Use scroll wheel to navigate slices \nImage dimensions: ({}, {}, {})'
                             .format(image_width, image_width, image_width))
                fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
                plt.show()

                plot_3d(ground_truth, mode, pred)

            # save prediction
            try:
                if with_mask:
                    os.mkdir('src/storage/datasets/with_mask/false_' + mode + '_segmentation/Sample' + str(i))
                else:
                    os.mkdir('src/storage/datasets/without_mask/false_' + mode + '_segmentation/Sample' + str(i))
            except OSError:
                print("Already existent")
            else:
                print("Successfully created.")
            if with_mask:
                np.save('src/storage/datasets/with_mask/false_' + mode + '_segmentation/Sample' + str(i) + '/pred_sigma' + str(sigma) + '_h' + str(h_value), pred)
            else:
                np.save('src/storage/datasets/without_mask/false_' + mode + '_segmentation/Sample' + str(i) + '/pred_sigma' + str(sigma) + '_h' + str(h_value), pred)

            # print(time.time() - start_time)
            times.append(time.time() - start_time)

        print("Mode: ", mode, "Mask: ", with_mask, "Foreground: ", only_foreground, "Sigma: ", sigma, "h_val: ", h_value)
        print("Best Time: ", np.min(times))
        print("Worst Time", np.max(times))
        print("Mean Time: ", np.mean(times))

        print("Best Rand Index: ", np.min(rand_indices))
        print("Worst Rand Index", np.max(rand_indices))
        print("Mean Rand Index: ", np.mean(rand_indices))
        print(rand_indices)

        print("Best Variation of Information: ", np.min(vois))
        print("Worst Variation of Information", np.max(vois))
        print("Mean Variation of Information: ", np.mean(vois))
        print(vois)

        print("Mean GT component number: ", np.mean(gt_component_nums))
        print("Mean Pred component number: ", np.mean(pred_component_nums))
