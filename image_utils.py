import matplotlib.pyplot as plt
import numpy

from utils import load_data

"""
kernel_size: The size of the Gaussian kernel, must be an odd integer
sigma: The sigma of the Gaussian applied to the input image
"""
def _create_gaussian_kernel(kernel_size, sigma):
    kernel = numpy.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i < (kernel_size + 1) / 2 and j < (kernel_size + 1) / 2:
                kernel[i, j] = numpy.exp(-((i + 0.5 - 0.5 * kernel_size) ** 2 + (j + 0.5 - 0.5 * kernel_size) ** 2) / (2 * sigma ** 2))
            elif j >= (kernel_size + 1) / 2:
                kernel[i, j] = kernel[i, kernel_size - 1 - j]
            else:
                kernel[i, j] = kernel[kernel_size - 1 - i, kernel_size - 1 - j]
    kernel /= numpy.sum(kernel)      
    return kernel

"""
I: The image, represented by a 2d array
"""
def gaussian_blur(I, kernel_size, sigma):
    nx = I.shape[0]
    ny = I.shape[1]
    kernel = _create_gaussian_kernel(kernel_size, sigma)
    kernel_center = (kernel_size - 1) / 2
    new_I = numpy.zeros((nx, ny))
    for x in range(nx):
        for y in range(ny):
            if (x - kernel_center < 0) or (x + kernel_center >= nx) or (y - kernel_center > 0) or (y + kernel_center >= ny):
                kernel_sum = 0
                for dx in range(-kernel_center, kernel_center + 1):
                    for dy in range(-kernel_center, kernel_center + 1):
                        if x + dx >= 0 and x + dx < nx and y + dy >= 0 and y + dy < ny:
                            new_I[x, y] += kernel[kernel_center + dx, kernel_center + dy] * I[x + dx, y + dy]
                            kernel_sum += kernel[kernel_center + dx, kernel_center + dy]
                new_I[x, y] /= kernel_sum
            else:
                for dx in range(-kernel_center, kernel_center + 1):
                    for dy in range(-kernel_center, kernel_center + 1):
                        new_I[x, y] += kernel[kernel_center + dx, kernel_center + dy] * I[x + dx, y + dy]
    return new_I

if __name__ == '__main__':
    print(_create_gaussian_kernel(3, 1))
    print("Loading data")
    Xtrain, Ytrain, Xtest = load_data()
    plt.imshow(Xtrain[0, :, :, 0], cmap='gray', interpolation='none')
    plt.show()
    blur1 = gaussian_blur(Xtrain[0, :, :, 0], kernel_size=5, sigma=1.6)
    plt.imshow(blur1, cmap='gray', interpolation='none')
    plt.show()
        
