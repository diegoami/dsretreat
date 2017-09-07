import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2


im = plt.imread('../../../../data/moonlanding.png').astype(float)

plt.figure()
plt.imshow(im, plt.cm.gray)
plt.title('Original image')
#
plt.show()

dft = fft2(im)
print(dft)
#
from matplotlib.colors import LogNorm
#
plt.imshow(np.abs(dft), norm=LogNorm(vmin=5))
plt.colorbar()
plt.show()


def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

keep_fraction = 0.1

im_fft2 = dft.copy()

r, c = im_fft2.shape

im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0

im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0

plt.figure()
plot_spectrum(im_fft2)
plt.title('Filtered Spectrum')
plt.show()


im_new = ifft2(im_fft2).real

plt.figure()
plt.imshow(im_new, plt.cm.gray)
plt.title('Reconstructed Image')
plt.show()