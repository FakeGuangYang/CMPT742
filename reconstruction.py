import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy


def reconstruction(source_img):
    # calculate height and width for the given source image
    height, width = source_img.shape
    k = height * width
    # initiate matrices - Av = b
    A = scipy.sparse.lil_matrix((k, k))  # sparse matrix A for second order derivatives of the output image
    source_img_sod = np.zeros((height, width)) # second order derivatives of the source image

    # Laplacian kernels for 3 different scenarios - middle, side and corner
    laplacian_kernel_c = np.array([[0, 1, 0],
                                   [1, -4, 1],
                                   [0, 1, 0]])
    laplacian_kernel_v = np.array([[0, 1, 0],
                                   [0, -2, 0],
                                   [0, 1, 0]])
    laplacian_kernel_h = np.array([[0, 0, 0],
                                   [1, -2, 1],
                                   [0, 0, 0]])
    # laplacian results for 3 different scenarios
    laplacian_c = scipy.signal.convolve2d(source_img, laplacian_kernel_c, mode='same') # corner
    laplacian_v = scipy.signal.convolve2d(source_img, laplacian_kernel_v, mode='same') # vertical
    laplacian_h = scipy.signal.convolve2d(source_img, laplacian_kernel_h, mode='same') # horizontal

    # calculate source image second order derivatives
    for i in range(height):
        for j in range(width):
            if 0 < i < height - 1 and 0 < j < width - 1:
                source_img_sod[i, j] = int(laplacian_c[i][j])
            elif i == 0 or i == height - 1:
                source_img_sod[i, j] = int(laplacian_h[i][j])
            elif j == 0 or j == width - 1:
                source_img_sod[i, j] = int(laplacian_v[i][j])
    # set 4 corner values to the source image value respectively
    source_img_sod[0][0] = source_img[0][0]
    source_img_sod[height - 1][0] = source_img[height - 1][0]
    source_img_sod[0][width - 1] = source_img[0][width - 1]
    source_img_sod[height - 1][width - 1] = source_img[height - 1][width - 1]
    # reshape to b
    b = source_img_sod.reshape((height * width, 1))

    # calculate A
    for i in range(k):
        # 4 corners
        if i == 0 or i == height - 1 or i == width * (height - 1) or i == k-1:
            A[i, i] = 1
        # 2 vertical sides
        elif 0 < i < width - 1 or width * (height - 1) < i < k - 1:
            A[i, i] = -2
            A[i, i - 1] = A[i, i + 1] = 1
        # 2 horizontal sides
        elif (0 < i < width * (height - 1) and i % width == 0) or (width - 1 < i < k - 1 and (i + 1) % width == 0):
            A[i, i] = -2
            A[i, i - width] = A[i, i + width] = 1
        # middle pixels
        else:
            A[i, i] = -4
            A[i, i - 1] = A[i, i + 1] = A[i, i - width] = A[i, i + width] = 1
    # use spsolve to calculate which is faster than linalg.lstsq
    v = scipy.sparse.linalg.spsolve(A, b)

    # reshape to output image and return
    output_image = v.reshape((height, width))

    # calculate least square error
    lse = ((A@v - b.T) @ (A@v - b.T).T) ** 0.5
    return output_image, lse[0][0]


if __name__ == '__main__':
    # read source image and turn to grey scale
    image_path = 'large.jpg'
    source_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # reconstruct image
    reconstructed_image, least_square_error = reconstruction(source_image)
    print("Least Square Error =", least_square_error)

    # use Matplotlib to present the source image and the reconstructed image
    plt.subplot(1, 2, 1)
    plt.imshow(source_image, cmap='gray')
    plt.title('Source Image')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title('Reconstructed Image')

    plt.show()

    # use OpenCV to save the reconstructed image
    cv2.imwrite('reconstructed_image.jpg', reconstructed_image)
