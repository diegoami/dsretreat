import numpy as np
from scipy import misc
face = misc.face(gray=True)  # 2D grayscale image
import pylab as plt
face = misc.face(gray=True)
plt.imshow(face)
plt.imshow(face, cmap=plt.cm.gray)
crop_face = face[100:-100, 100:-100]
sy, sx = face.shape
y, x = np.ogrid[0:sy, 0:sx] # x and y indices of pixels

centerx, centery = (600, 700) # center of the image
mask = ((y*2 - centery)**2 + (x - centerx)**2) > 300**2 # circle
face[mask] = 0
plt.imshow(face)
plt.show()
