from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']

idx = np.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]
matplotlib.use("agg")


def classify(images, label_vector, query, k):
    """
    Question 2.a
    Given a set of training images, a label vector, a query image and an integer k,
    outputs a prediction on what label the query image belongs to. Classifies the query image based on its
    k-nearest neighbors in the dataset
    :param images: Set of training images
    :param label_vector: Vector of labels
    :param query: Image to be classified
    :param k: Integer
    :return: Predicted label of the given query
    :rtype: str
    """
    distances = np.array([np.linalg.norm(img - query) for img in images])  # An array consisting of the distances
    # between any image in the dataset and the query image. The distance is simply the norm of the difference vector.
    k_nearest_idxs = np.argpartition(distances, k)[:k]  # An array of length k of the smallest elements in
    # the distances array.
    counts = np.bincount(label_vector[k_nearest_idxs].astype(int))  # Counts the occurrences of the labels belonging
    # to the elements with the indexes from k_nearest_idxs.
    return str(np.argmax(counts))  # Returns the label with the most occurrences


def accuracy(n, k):
    """
    Question 2.b
    Runs the classify function on n images from the training data, and returns the accuracy of the algorithm
    compared to the actual labels
    :param n: Number of images to run
    :param k: parameter for the classify function
    :return: Accuracy of the classify function
    :rtype: float
    """
    current_train = train[:n]
    current_labels = train_labels[:n]
    # Copies of the first n elements of train and train_labels

    predictions = np.array([classify(current_train, current_labels, img, k) == test_labels[i]
                            for i, img in enumerate(test)])
    #  An array of booleans. If classify() labeled the image correctly then the element in the array will be True,
    # otherwise it will be False.
    return np.count_nonzero(predictions)/predictions.size


def iterate_on_k():
    """
    Question 2.c
    Plots the prediction accuracy as a function of k, for k=1,...,100, and n=1000
    :return: None
    """
    n = 1000
    k = 10
    print(f"(b) Accuracy of prediction:{accuracy(n,k)}")

    acc_array = np.array([accuracy(1000, k) for k in range(1, 101)])  # Array of results from accuracy() for 1<=k<=100
    k_array = list(range(1, 101))
    plt.plot(k_array, acc_array)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title('Prediction Accuracy As a Function of k')
    #plt.savefig('Prediction accuracy as a function of k.pdf')
    plt.show()
    return


def iterate_on_n():
    """
    Question 2.d
    Plots the prediction accuracy as a function of n, for n=100,200,...,5000, and k=1
    :return:
    """
    acc_array = np.array([accuracy(n, 1) for n in range(100, 5001, 100)])
    n_array = list(range(100, 5001, 100))
    plt.plot(n_array, acc_array)
    plt.xlabel('n')
    plt.ylabel('Accuracy')
    plt.title('Prediction Accuracy As a Function of n')
    #plt.savefig('Prediction accuracy as a function of n.pdf')
    plt.show()


iterate_on_k()
iterate_on_n()

