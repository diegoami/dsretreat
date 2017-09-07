import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy import ndimage
from scipy import signal


im = plt.imread('moonlanding.png').astype(float)
from matplotlib import pyplot as plt
fig, axes = plt.subplots(2, 2)
import numpy as np

orig_face = np.copy(im).astype(np.float)

noisy_face = orig_face +  im.std() * 0.5 * np.random.standard_normal(im.shape)
blurred_face = ndimage.gaussian_filter(noisy_face, sigma=3)
median_face = ndimage.median_filter(noisy_face, size=5)
wiener_face = signal.wiener(noisy_face, (5, 5))
axes[0,0].imshow(noisy_face, cmap=plt.cm.gray)
axes[0,1].imshow(blurred_face, cmap=plt.cm.gray)
axes[1,0].imshow(median_face, cmap=plt.cm.gray)
axes[1,1].imshow(wiener_face, cmap=plt.cm.gray)
plt.show()