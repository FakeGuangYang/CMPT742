import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os


def find_matching_keypoints(image1, image2):
    # Input: two images (numpy arrays)
    # Output: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2


def drawlines(img1, img2, lines, pts1, pts2):
    # img1: image on which we draw the epilines for the points in img2
    # lines: corresponding epilines
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def FindFundamentalMatrix(pts1, pts2):
    # Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    # Output: fundamental matrix (numpy array of shape (3, 3))

    # todo: Normalize the points
    def normalize_points(pts):
        # calculate the centroid of the points
        centroid = np.mean(pts, axis=0)
        # calculate the mean distance of all points to the centroid
        mean_distance_of_all_points_to_centroid = np.mean(np.linalg.norm(pts - centroid, axis=1))
        # make the mean distance of the points to the centroid equal to √2 (mean * scale = √2)
        scale = np.sqrt(2) / mean_distance_of_all_points_to_centroid
        # construct the transformation matrix
        M = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        # point normalization
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        normalized_pts = (M @ pts_h.T).T
        return normalized_pts, M

    normalized_pts1, M1 = normalize_points(pts1)
    normalized_pts2, M2 = normalize_points(pts2)

    # todo: Form the matrix A
    # form an A matrix with n paired points
    len_normalized_pts1 = normalized_pts1.shape[0]
    A = np.zeros((len_normalized_pts1, 9))
    for i in range(len_normalized_pts1):
        x1, y1, _ = normalized_pts1[i]
        x2, y2, _ = normalized_pts2[i]
        A[i] = [x1 * x2, x2 * y1, x2, y2 * x1, y1 * y2, y2, x1, y1, 1]

    # todo: Find the fundamental matrix
    # calculate SVD for A_T*A and solve for F from V_T
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint on F and set smallest singular value to 0
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    # Denormalize F
    F = M2.T @ F @ M1
    F = F / F[-1, -1]

    return F


def FindFundamentalMatrixRansac(pts1, pts2, num_trials=1000, threshold=0.01):
    # Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    # Output: fundamental matrix (numpy array of shape (3, 3))

    # todo: Run RANSAC and find the best fundamental matrix
    best_F = None
    max_inliers = 0
    best_inliers_mask = None  # To store the inlier mask for the best model

    for _ in range(num_trials):
        # Randomly sample 8 points
        indices = np.random.choice(len(pts1), 8, replace=False)
        sampled_pts1 = pts1[indices]
        sampled_pts2 = pts2[indices]

        # Compute F using the 8-point algorithm
        try:
            F = FindFundamentalMatrix(sampled_pts1, sampled_pts2)
        except np.linalg.LinAlgError:
            continue

        # Compute Sampson distance for all points
        pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

        # Epipolar line in the second image for points in the first image
        l2 = F @ pts1_h.T  # l2 = (a, b, c) for each point
        l2 /= np.sqrt(l2[0] ** 2 + l2[1] ** 2)  # Normalize

        # Compute the distance from points in the second image to the epipolar line
        dist2 = np.abs(np.sum(l2.T * pts2_h, axis=1))

        # Count inliers
        inliers_mask = dist2 < threshold
        num_inliers = np.sum(inliers_mask)

        # Update the best model if more inliers are found
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_F = F
            best_inliers_mask = inliers_mask

            # Early stopping if inliers are sufficient
            if max_inliers > 0.90 * len(pts1):
                break

    # Refine F using all inliers
    if best_inliers_mask is not None:
        inlier_pts1 = pts1[best_inliers_mask]
        inlier_pts2 = pts2[best_inliers_mask]
        best_F = FindFundamentalMatrix(inlier_pts1, inlier_pts2)

    return best_F


if __name__ == '__main__':
    # Set parameters
    data_path = './data'
    use_ransac = False

    # Load images
    image1_path = os.path.join(data_path, 'notredam1.jpg')
    image2_path = os.path.join(data_path, 'notredam2.jpg')
    image1 = np.array(Image.open(image1_path).convert('L'))
    image2 = np.array(Image.open(image2_path).convert('L'))

    # Find matching keypoints
    pts1, pts2 = find_matching_keypoints(image1, image2)

    # Builtin opencv function for comparison
    F_true = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]
    F_ransac = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)[0]
    print("F_true: ", F_true)
    print("F_ransac: ", F_ransac)

    # todo: FindFundamentalMatrix
    if use_ransac:
        F = FindFundamentalMatrixRansac(pts1, pts2)
    else:
        F = FindFundamentalMatrix(pts1, pts2)
    print(F)
    print("difference: ", abs(F_ransac - F))
    # Find epilines corresponding to points in second image,  and draw the lines on first image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F_ransac)
    lines1 = lines1.reshape(-1, 3)
    img1_left, img2 = drawlines(image1, image2, lines1, pts1, pts2)
    fig, axis = plt.subplots(1, 2)

    axis[0].imshow(img1_left)
    axis[0].set_title('Image 1')
    axis[0].axis('off')
    axis[1].imshow(img2)
    axis[1].set_title('Image 2')
    axis[1].axis('off')

    plt.show()

    # Find epilines corresponding to points in first image, and draw the lines on second image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F_ransac)
    lines2 = lines2.reshape(-1, 3)
    img1_right, img2 = drawlines(image2, image1, lines2, pts2, pts1)
    fig, axis = plt.subplots(1, 2)

    axis[0].imshow(img1_right)
    axis[0].set_title('Image 1')
    axis[0].axis('off')
    axis[1].imshow(img2)
    axis[1].set_title('Image 2')
    axis[1].axis('off')

    plt.show()

    # Show the combined image
    fig, axis = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 2]})
    axis[0].imshow(img1_left)

    axis[0].axis('off')
    axis[1].imshow(img1_right)

    axis[1].axis('off')
    plt.tight_layout(pad=0)
    plt.show()
