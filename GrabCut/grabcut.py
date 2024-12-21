import numpy as np
import cv2
import argparse
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from igraph import Graph
import time

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel

# Added global variables
prev_energy = 0
THRESHOLD = 1500

# Define the GrabCut algorithm function
def grabcut(img, rect, n_components=5):
    # Assign initial labels to the pixels based on the bounding box
    img = np.asarray(img, dtype=np.float64)  # Adding this line fixed an issue with noise 
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Adding these two lines made it so the rectangle wouldn't go to the bottom right edge of the image
    h -= y
    w -= x

    # Initalize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initialize_GMMs(img, mask, n_components)

    num_iters = 1000
    for i in range(num_iters):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Finalize the mask, assign probable fg/bg pixels to definite fg/bg
    mask[mask == GC_PR_FGD] = GC_FGD
    mask[mask == GC_PR_BGD] = GC_BGD
    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initialize_GMMs(img, mask, n_components=5):
    """
    Initializes Gaussian Mixture Models (GMMs) for background and foreground using K-Means clustering.

    Args:
        img (np.ndarray): The input image as a 3D numpy array (H, W, C).
        mask (np.ndarray): The mask defining pixel labels for background and foreground.
        n_components (int, optional): Number of components in the GMM. Defaults to 5.

    Returns:
        tuple: A tuple containing initialized background GMM and foreground GMM.
    """
    # Separate background and foreground pixels
    bg_pixels = img[(mask == GC_PR_BGD) | (mask == GC_BGD)].reshape(-1, 3)
    fg_pixels = img[(mask == GC_PR_FGD) | (mask == GC_FGD)].reshape(-1, 3)

    # Initialize GMMs using Kmeans
    bg_Kmeans = KMeans(n_clusters=n_components)
    bg_Kmeans.fit(bg_pixels)
    fg_Kmeans = KMeans(n_clusters=n_components)
    fg_Kmeans.fit(fg_pixels)

    # Initialize GMMs
    bgGMM = GaussianMixture(n_components=n_components, means_init=bg_Kmeans.cluster_centers_)
    fgGMM = GaussianMixture(n_components=n_components, means_init=fg_Kmeans.cluster_centers_)

    # Fit the GMMs
    bgGMM.fit(bg_pixels)
    fgGMM.fit(fg_pixels)

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    """
    Updates the parameters of the background and foreground GMMs based on the current pixel classification.

    Args:
        img (np.ndarray): The input image as a 3D numpy array (H, W, C).
        mask (np.ndarray): The mask defining pixel labels for background and foreground.
        bgGMM (GaussianMixture): The GMM representing the background.
        fgGMM (GaussianMixture): The GMM representing the foreground.

    Returns:
        tuple: Updated background GMM and foreground GMM.
    """
    # Separate background and foreground pixels
    bg_pixels = img[(mask == GC_PR_BGD) | (mask == GC_BGD)].reshape(-1, 3)
    fg_pixels = img[(mask == GC_PR_FGD) | (mask == GC_FGD)].reshape(-1, 3)

    # Update GMM parameters
    bgGMM = update_parameters(bgGMM, bg_pixels)
    fgGMM = update_parameters(fgGMM, fg_pixels)

    # Remove zero weights if they exist
    bgGMM = remove_zero_weights(bgGMM)
    fgGMM = remove_zero_weights(fgGMM)

    return bgGMM, fgGMM


def update_parameters(GMM: GaussianMixture, pixels) -> GaussianMixture:
    """
    Updates the parameters (means, weights, covariances) of a Gaussian Mixture Model.

    Args:
        GMM (GaussianMixture): The GMM to update.
        pixels (np.ndarray): The pixel data assigned to the GMM.

    Returns:
        GaussianMixture: The updated GMM.
    """
    # Initialize arrays to store updated GMM parameters
    n_components = GMM.n_components
    covars = np.zeros((n_components, 3, 3))
    means = np.zeros((n_components, 3))
    weights = np.zeros(n_components)

    # Iterate over the GMM components, calculate the mean and covariance of each component,
    # then update the GMM weights, means and covariances
    for i in range(n_components):
        # Get the pixels that belong to the current component
        component_mask = GMM.predict(pixels) == i
        component_data = pixels[component_mask]

        # Update the weights, means and covariances if the component has data
        if len(component_data) > 0:
            weights[i] = len(component_data) / len(pixels)  # Fraction of total pixels
            covar, mean = cv2.calcCovarMatrix(component_data, None,
                                               cv2.COVAR_NORMAL | cv2.COVAR_SCALE | cv2.COVAR_ROWS)
            means[i] = mean.flatten()
            covars[i] = covar

    # Assign updated parameters to the GMM
    GMM.weights_ = weights
    GMM.means_ = means
    GMM.covariances_ = covars

    return GMM


def remove_zero_weights(GMM: GaussianMixture) -> GaussianMixture:
    """
    Removes components with weights smaller than some epsilon from a Gaussian Mixture Model.

    Args:
        GMM (GaussianMixture): The Gaussian Mixture Model to modify.

    Returns:
        GaussianMixture: The modified GMM with zero-weight components removed.
    """

    epsilon = 1e-6
    # Identify components with weights smaller than epsilon
    zero_weight_indices = np.where(GMM.weights_ < epsilon)[0]

    # If no components to remove, return the GMM as-is
    if len(zero_weight_indices) == 0:
        return GMM

    # Remove components from weights, means, and covariances
    GMM.weights_ = np.delete(GMM.weights_, zero_weight_indices, axis=0)
    GMM.means_ = np.delete(GMM.means_, zero_weight_indices, axis=0)
    GMM.covariances_ = np.delete(GMM.covariances_, zero_weight_indices, axis=0)

    # Recompute precisions and precision cholesky
    GMM.precisions_ = np.array([np.linalg.inv(cov) for cov in GMM.covariances_])
    GMM.precisions_cholesky_ = np.array([np.linalg.cholesky(prec) for prec in GMM.precisions_])

    # Update the number of components
    GMM.n_components = len(GMM.weights_)

    return GMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    """
    Calculates the minimum cut for segmenting the image into background and foreground using GraphCut.

    Args:
        img (np.ndarray): The input image as a 3D numpy array (H, W, C).
        mask (np.ndarray): The current pixel label mask.
        bgGMM (GaussianMixture): The GMM representing the background.
        fgGMM (GaussianMixture): The GMM representing the foreground.

    Returns:
        tuple: A tuple containing the min-cut sets and the energy value.
    """
    img_size = img.shape[0] * img.shape[1]

    # Initialize empty graph with img_size + 2 nodes
    g = Graph(img_size + 2, directed=False)
    source = img_size  # Background node
    target = img_size + 1  # Foreground node

    # Flatten image and mask for ease of manipulation
    two_dim_img = img.reshape(img_size, img.shape[2])
    pixels_id = np.arange(img_size).reshape(img.shape[0], img.shape[1])
    two_dim_mask = mask.reshape(-1)

    # Calculate N-Links
    cache = {}
    nlinks_edges, nlinks_capacities, beta = calculate_nlinks(img, two_dim_img, pixels_id, cache)

    # Calculate T-Links
    K = np.max(nlinks_capacities)
    tlinks_edges, tlinks_capacities = calculate_tlinks(two_dim_img, two_dim_mask, bgGMM, fgGMM, K)

    # Add edges and capacities to the graph
    edges = np.concatenate((nlinks_edges, tlinks_edges)).tolist()
    capacities = np.concatenate((nlinks_capacities, tlinks_capacities)).tolist()
    g.add_edges(edges)

    # Perform mincut and partition the graph
    g_min_cut = g.mincut(source, target, capacities)
    bg = g_min_cut.partition[0]
    fg = g_min_cut.partition[1]
    bg.remove(source)
    fg.remove(target)

    min_cut = [bg, fg]
    energy = g_min_cut.value

    return min_cut, energy


def calculate_nlinks(img, two_dim_img, pixels_id, cache):
    """
    Calculates the N-links for constructing the graph.

    Args:
        img (np.ndarray): The input image as a 3D numpy array (H, W, C).
        two_dim_img (np.ndarray): A reshaped version of the image for easier pixel access.
        pixels_id (np.ndarray): The array of pixel IDs in the reshaped image.
        cache (dict): Cache to store precomputed N-links for efficiency.

    Returns:
        tuple: N-links edges, their capacities, and the beta parameter.
    """
    if len(cache) == 0:  # Cache N-Links to avoid computing more than once
        beta = calc_beta(img)

        # Define adjacency weights for neighboring pixels
        def N_adjacent(n, m): return 50 * np.exp(-beta * np.linalg.norm(two_dim_img[n] - two_dim_img[m]) ** 2)

        def N_diagonal(n, m): return (50 / 2 ** 0.5) * np.exp(
            -beta * np.linalg.norm(two_dim_img[n] - two_dim_img[m]) ** 2)

        # Calculate edges and capacities for adjacent and diagonal neighbors
        top_left_edges = np.column_stack((pixels_id[1:, 1:].reshape(-1), pixels_id[:-1, :-1].reshape(-1)))
        top_left_capacities = [N_diagonal(n, m) for n, m in top_left_edges]

        top_edges = np.column_stack((pixels_id[1:, :].reshape(-1), pixels_id[:-1, :].reshape(-1)))
        top_capacities = [N_adjacent(n, m) for n, m in top_edges]

        top_right = np.column_stack((pixels_id[1:, :-1].reshape(-1), pixels_id[:-1, 1:].reshape(-1)))
        top_right_capacities = [N_diagonal(n, m) for n, m in top_right]

        left_edges = np.column_stack((pixels_id[:, 1:].reshape(-1), pixels_id[:, :-1].reshape(-1)))
        left_capacities = [N_adjacent(n, m) for n, m in left_edges]

        nlinks_edges = np.concatenate((top_left_edges, top_edges, top_right, left_edges))
        nlinks_capacities = np.concatenate((top_left_capacities, top_capacities, top_right_capacities, left_capacities))

        cache[0] = (nlinks_edges, nlinks_capacities, beta)

    return cache[0]


def calc_beta(img):
    """
    Computes the beta parameter, which normalizes intensity differences between neighboring pixels.

    Args:
        img (np.ndarray): The input image as a 3D numpy array (H, W, C).

    Returns:
        float: The computed beta value.
    """
    top_left_dist = np.sum((img[1:, 1:] - img[:-1, :-1]) ** 2)  # sum of img_size - img.shape[0] - img.shape[1] elements
    top_dist = np.sum((img[1:, :] - img[:-1, :]) ** 2)  # sum of img_size - img.shape[0] elements
    top_right_dist = np.sum(
        (img[1:, :-1] - img[:-1, 1:]) ** 2)  # sum of img_size - img.shape[0] - img.shape[1] elements
    left_dist = np.sum((img[:, 1:] - img[:, :-1]) ** 2)  # sum of img_size - img.shape[1] elements

    img_size = img.shape[0] * img.shape[1]
    total_elements_summed = 4 * img_size - 3 * (img.shape[0] + img.shape[1])
    expected_dist = (left_dist + top_left_dist + top_dist + top_right_dist) / total_elements_summed

    beta = 1 / (2 * expected_dist)
    return beta


def calculate_tlinks(two_dim_img, two_dim_mask, bgGMM, fgGMM, K):
    """
    Calculates the T-links for the graph.

    Args:
        two_dim_img (np.ndarray): A reshaped version of the image for easier pixel access.
        two_dim_mask (np.ndarray): A reshaped version of the pixel label mask.
        bgGMM (GaussianMixture): The GMM representing the background.
        fgGMM (GaussianMixture): The GMM representing the foreground.
        K (float): Maximum capacity value for T-links.

    Returns:
        tuple: T-links edges and their capacities.
    """
    # Identify pixels labeled as background, foreground and probable background/foreground
    bg_indices = np.where(two_dim_mask == GC_BGD)[0]
    fg_indices = np.where(two_dim_mask == GC_FGD)[0]
    pr_bg_indices = np.where(two_dim_mask == GC_PR_BGD)[0]
    pr_fg_indices = np.where(two_dim_mask == GC_PR_FGD)[0]

    source = len(two_dim_img)  # Source node index - aka background node
    target = len(two_dim_img) + 1  # Target node index - aka foreground node

    def D_fore_back(edges, gmm):
        # Compute D-term for edges connected to source/target
        indexes_connected_to_st = np.array(edges)[:, 1]  # Extract the indices of the target nodes
        p_to_st = two_dim_img[indexes_connected_to_st]  # Extract the pixel values for these indices

        log_likelihood = gmm.score_samples(p_to_st)  # Log likelihood for each pixel
        D = -log_likelihood  # Negative log likelihood for the data term

        return D

    # T-links for definite background and foreground
    source_to_bg_edges = np.column_stack(([source] * len(bg_indices), bg_indices))
    source_to_bg_cap = [K] * len(source_to_bg_edges)

    target_to_fg_edges = np.column_stack(([target] * len(fg_indices), fg_indices))
    target_to_fg_cap = [K] * len(target_to_fg_edges)

    # T-links for probable background and foreground, based on GMM
    source_to_pr_fg_edges = np.column_stack(([source] * len(pr_fg_indices), pr_fg_indices))
    source_to_pr_fg_cap = D_fore_back(source_to_pr_fg_edges, fgGMM)

    target_to_pr_fg_edges = np.column_stack(([target] * len(pr_fg_indices), pr_fg_indices))
    target_to_pr_fg_cap = D_fore_back(target_to_pr_fg_edges, bgGMM)

    # Combine all T-links edges and capacities
    tlinks_edges = np.concatenate(
        [source_to_bg_edges, source_to_pr_fg_edges, target_to_fg_edges, target_to_pr_fg_edges]).astype(int)
    tlinks_capacities = np.concatenate([source_to_bg_cap, source_to_pr_fg_cap, target_to_fg_cap, target_to_pr_fg_cap])

    # Handle probable background pixels
    if len(pr_bg_indices) != 0:
        source_to_pr_bg_edges = np.column_stack(([source] * len(pr_bg_indices), pr_bg_indices))
        source_to_pr_bg_cap = D_fore_back(source_to_pr_bg_edges, fgGMM)

        target_to_pr_bg_edges = np.column_stack(([target] * len(pr_bg_indices), pr_bg_indices))
        target_to_pr_bg_cap = D_fore_back(target_to_pr_bg_edges, bgGMM)

        tlinks_edges = np.concatenate((tlinks_edges, source_to_pr_bg_edges, target_to_pr_bg_edges))
        tlinks_capacities = np.concatenate((tlinks_capacities, source_to_pr_bg_cap, target_to_pr_bg_cap))

    return tlinks_edges, tlinks_capacities


def update_mask(mincut_sets, mask):
    """
    Updates the mask labels based on the result of the min-cut.

    Args:
        mincut_sets (list): Partitioned sets from the min-cut operation.
        mask (np.ndarray): The current pixel label mask.

    Returns:
        np.ndarray: The updated mask.
    """
    two_dim_mask = mask.reshape(-1)  # Flatten mask for easier indexing

    # Update background pixels - mark as definite or probable background
    new_bg = two_dim_mask[mincut_sets[0]]
    new_bg[new_bg != GC_BGD] = GC_PR_BGD

    # Update foreground pixels - mark as definite or probable foreground
    new_fg = two_dim_mask[mincut_sets[1]]
    new_fg[new_fg != GC_FGD] = GC_PR_FGD

    # Apply updates to the flat mask, then reshape back to original dimensions
    two_dim_mask[mincut_sets[0]] = new_bg
    two_dim_mask[mincut_sets[1]] = new_fg
    mask = two_dim_mask.reshape(mask.shape[0], mask.shape[1])

    return mask


def check_convergence(energy):
    """
    Checks if the algorithm has converged based on the energy value.

    Args:
        energy (float): The current energy of the graph.

    Returns:
        bool: True if convergence criteria are met, False otherwise.
    """
    global prev_energy
    global mask
    threshold = THRESHOLD  # Convergence threshold for energy difference
    is_conv = False
    # Check if energy change is below the threshold
    if np.abs(energy - prev_energy) <= threshold:
        is_conv = True
    prev_energy = energy  # Update prev_energy for the next iteration
    return is_conv


def cal_metric(predicted_mask, gt_mask):
    """
    Calculates the accuracy and Jaccard similarity metric between the predicted mask and ground truth.

    Args:
        predicted_mask (np.ndarray): The predicted segmentation mask.
        gt_mask (np.ndarray): The ground truth segmentation mask.

    Returns:
        tuple: Accuracy and Jaccard similarity index.
    """
    # Identify foreground pixels in both masks
    fg_predicted = predicted_mask[predicted_mask == 1]
    fg_gt = gt_mask[gt_mask == 1]

    # Compute pixel-wise accuracy
    accuracy = np.sum(predicted_mask == gt_mask) / gt_mask.size

    # Compute Jaccard similarity
    cap = gt_mask[(gt_mask == 1) & (predicted_mask == 1)]
    jaccard = cap.size / (fg_predicted.size + fg_gt.size - cap.size)

    return accuracy, jaccard


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='fullmoon', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    start_time = time.time()
    mask, bgGMM, fgGMM = grabcut(img, rect)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time = {elapsed_time:.2f}s")
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
