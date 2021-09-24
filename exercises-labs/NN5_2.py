import numpy as np
import matplotlib.pyplot as plt

#f = open('train-images.idx3-ubyte','rb')
f = open('t10k-images.idx3-ubyte','rb')

imSize=28
imDim=imSize**2
#N=1

f.read(16)
buffer=f.read(imDim)
f.close()

data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
im=data.reshape(imSize,imSize)
plt.imshow(im)
plt.show()
plt.figure()
plt.imshow(255-im,cmap='gray')
plt.show()

#f = open('train-labels.idx1-ubyte','rb')
f = open('t10k-labels.idx1-ubyte','rb')

N=1
f.read(8)
buffer=f.read(1)
f.close()

target = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)
print(target)