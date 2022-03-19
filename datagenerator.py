"""Data Generation."""
import numpy as np


def rotation_matrix(theta: float) -> np.ndarray:
    """Create rotation matrix."""
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])


def generate_data() -> np.ndarray:
    """Generate data."""
    np.random.seed(10)

    # Different means
    mu0 = np.array([0, 0])
    mu1 = np.array([2, 0])
    mu2 = np.array([8, 0])
    mu3 = np.array([10, 0])

    # Common covariance
    covariance = np.array([[5, 0], [0, 1]]) @ rotation_matrix(np.pi / 4)

    # Core assumption of LDA
    points_x0 = np.random.multivariate_normal(mu0, covariance, size=100)
    points_y0 = np.random.multivariate_normal(mu1, covariance, size=100)
    points_x1 = np.random.multivariate_normal(mu2, covariance, size=100)
    points_y1 = np.random.multivariate_normal(mu3, covariance, size=100)

    # Create two cluster of data
    class_0 = np.vstack((points_x0, points_y0))
    class_1 = np.vstack((points_x1, points_y1))

    zero = np.zeros((class_0.shape[0], 1), dtype=int)
    one = np.ones((class_0.shape[0], 1), dtype=int)

    cluster_0 = np.hstack((class_0, zero))
    cluster_1 = np.hstack((class_1, one))

    features = np.vstack((cluster_0, cluster_1))
    samples = np.random.permutation(features)
    return samples

