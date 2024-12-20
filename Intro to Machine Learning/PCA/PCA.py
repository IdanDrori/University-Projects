import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np

matplotlib.use('TkAgg')


def plot_vector_as_image(image, h, w):
	"""
	utility function to plot a vector as image.
	Args:
	image - vector of pixels
	h, w - dimensions of original pi
	"""
	plt.imshow(image.reshape((h, w)), cmap="gray")
	plt.title('title', size=12)
	plt.show()


def get_pictures_by_name(name='Ariel Sharon'):
	"""
	Given a name returns all the pictures of the person with this specific name.
	YOU CAN CHANGE THIS FUNCTION!
	THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
	"""
	lfw_people = load_data()
	selected_images = []
	n_samples, h, w = lfw_people.images.shape
	target_label = list(lfw_people.target_names).index(name)
	for image, target in zip(lfw_people.images, lfw_people.target):
		if target == target_label:
			image_vector = image.reshape((h * w, 1))
			selected_images.append(image_vector)
	return np.array(selected_images).squeeze(axis=2), h, w


def load_data():
	# Don't change the resize factor!!!
	lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
	return lfw_people


def PCA(X, k):
	"""
	Compute PCA on the given matrix.

	Args:
		X - Matrix of dimensions (n,d). Where n is the number of sample points and d is the dimension of each sample.
		For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimension of the matrix would be (10,100).
		k - number of eigenvectors to return

	Returns:
		U- Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
			of the covariance matrix.
		S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
	"""
	_, _, S, U = PCA_with_SVD(X, k)  # The eigenvectors of the covariance matrix that correspond to the largest k
	# eigenvalues are equivalent to V_k from the SVD of matrix X. And the k largest eigenvalues of the covariance
	# matrix are the largest k singular values squared and normalized
	return U, np.square(S) * (1 / X.shape[0])


def PCA_with_SVD(X, k):
	"""
	Computes the PCA of the matrix X, uses SVD
	Args:
		X (np.ndarray): Data matrix of size (n,d)
		k (int): number of dimensions to reduce to

	Returns:
		principal_comp (np.ndarray): Top k principal components
		U (np.ndarray): Unitary matrix of size (n,k), its columns are the left singular vectors
		S (np.ndarray): Vector of size (k,1) of singular values in descending order
		V (np.ndarray): Unitary matrix of size (d,k), its columns are the right singular vectors
	"""
	X = X - np.mean(X, axis=0)  # Centering X
	U, S, Vt = np.linalg.svd(X)
	print(f"U.shape={U.shape}, S.shape={S.shape}, Vt.shape={Vt.shape}")
	# Select the top k components
	U_k = U[:, :k]
	S_k = S[:k]
	Vt_k = Vt[:k, :]
	principal_comps = np.matmul(U_k, np.diag(S_k))
	return principal_comps, U_k, S_k, Vt_k


def question_b(name):
	"""
  Runs PCA on matrix X whose rows are the flattened images of (name), and plots each of the eigenvectors as pictures
	Args:
		name (str): String of a person's name in the database

	Returns: None
	"""
	X, h, w = get_pictures_by_name(name)
	U, S = PCA(X, 10)
	fig = plt.figure()
	for i in range(1, 11):
		fig.add_subplot(2, 5, i)
		plt.imshow(U[i - 1].reshape((h, w)), cmap='gray')
		plt.title(f'Vector {i}', size=12)
	fig.tight_layout()
	plt.show()


def question_c(name):
	"""
  Select 5 pictures at random of (name), and plot for different values of k, the original pictures next to the pictures 
  obtained by transforming the reduced pictures back to their original dimension.
	Args:
		name (str): String of a person's name in the database

	Returns: None

	"""
	k_list = [1, 5, 10, 30, 50, 100]
	X, h, w = get_pictures_by_name(name)
	random_pict_idxs = np.random.randint(0, X.shape[0], size=5)  # The indexes of 5 random pictures from the matrix X
	l2_sums = []
	print(f"X.shape={X.shape}")
	for k in k_list:
		cur_l2 = 0
		principal_comps, U_k, S_k, Vt_k = PCA_with_SVD(X, k)
		print(f"principal_comps.shape={principal_comps.shape}, U_k.shape={U_k.shape}, S_k.shape={S_k.shape}, Vt_k.shape={Vt_k.shape}")
		A = np.matmul(Vt_k, X.T)
		X_reconstructed = np.matmul(Vt_k.T, A).T
		fig = plt.figure(figsize=(4, 8))
		for i, img_idx in enumerate(random_pict_idxs):
			fig.add_subplot(5, 2, 2 * i + 1)
			plt.imshow(X[img_idx].reshape((h, w)), cmap='gray')
			fig.add_subplot(5, 2, 2 * i + 2)
			plt.imshow(X_reconstructed[img_idx].reshape((h, w)), cmap='gray')
			cur_l2 += np.linalg.norm(X[img_idx] - X_reconstructed[img_idx])
		l2_sums.append(cur_l2)
		plt.suptitle(f"k = {k}, Original : Transformed")
		plt.show()

	plt.plot(k_list, l2_sums)
	plt.title('l_2 Distances as a Function of k')
	plt.xlabel('Number of Principal Components (k)')
	plt.ylabel('Sum of l_2 Distances')
	plt.grid(True)
	plt.show()


if __name__ == '__main__':
	question_b("George W Bush")
	question_c("George W Bush")
