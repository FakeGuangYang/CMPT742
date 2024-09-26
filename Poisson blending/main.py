import cv2
import numpy as np
import scipy
from align_target import align_target


# main poisson blending function
def poisson_blending(source_img, target_img, mask):
    # split images into 3 channels (blue, green, red)
    source_b, source_g, source_r = cv2.split(source_img)
    target_b, target_g, target_r = cv2.split(target_img)

    # poisson blending for each channel
    r = poisson_blend_single_channel(source_r, target_r, mask)
    g = poisson_blend_single_channel(source_g, target_g, mask)
    b = poisson_blend_single_channel(source_b, target_b, mask)

    # put the result into 0-255
    result_b = np.clip(b, 0, 255).astype(np.uint8)
    result_g = np.clip(g, 0, 255).astype(np.uint8)
    result_r = np.clip(r, 0, 255).astype(np.uint8)

    # merge three channels into one to get RGB image
    result_img = cv2.merge((result_b, result_g, result_r))

    return result_img


# poisson blending for single channel
def poisson_blend_single_channel(source_channel, target_channel, mask):
    # set up parameters
    height, width = mask.shape
    k = height * width
    A = scipy.sparse.lil_matrix((k, k))  # set up sparse matrix
    b = np.zeros(k)

    # laplacian for source pic
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    laplacian_source_channel = scipy.signal.convolve2d(source_channel, laplacian_kernel, mode='same')

    # calculate each pixel
    for y in range(height):
        for x in range(width):
            # pixels outside the patch
            if mask[y, x] == 0:
                b[y * width + x] = target_channel[y, x]
                A[y * width + x, y * width + x] = 1
            # pixels inside the patch
            else:
                # edges of the patch
                if mask[y - 1, x] == 0 or mask[y + 1, x] == 0 or mask[y, x - 1] == 0 or mask[y, x + 1] == 0:
                    b[y * width + x] = target_channel[y, x]
                    A[y * width + x, y * width + x] = 1
                # middle of the patch
                else:
                    b[y * width + x] = laplacian_source_channel[y,x]
                    A[y * width + x, y * width + x] = -4
                    A[y * width + x, (y - 1) * width + x] = A[y * width + x, (y + 1) * width + x] \
                        = A[y * width + x, y * width + (x - 1)] = A[y * width + x, y * width + (x + 1)] = 1

    # calculate f in A * f = b
    f = scipy.sparse.linalg.spsolve(A, b.T)

    # reshape the result channel into (height * weight)
    result_channel = f.reshape(height, width)
    return result_channel


if __name__ == '__main__':
    # read source and target images
    source_path = 'source1.jpg'
    target_path = 'target.jpg'
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    # align target image
    im_source, target_mask = align_target(source_image, target_image)
    kernel = np.ones(3*3, np.uint8)
    eroded_mask = cv2.erode(target_mask, kernel, iterations=1)

    # poisson blending
    blended_image = poisson_blending(im_source, target_image, eroded_mask)

    # show both source image and blended image
    # cv2.imwrite('blended.jpg', blended_image)
    cv2.imshow('blended image:', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
