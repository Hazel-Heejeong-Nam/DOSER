# Enhanced-open-set-recognition-with-disentanglement

On going class project of EEE4423 in Yonsei university.

download MNIST dataset
``` python
!wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
!wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
!wget --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
!wget --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
!gzip -f -d train-images-idx3-ubyte.gz
!gzip -f -d train-labels-idx1-ubyte.gz
!gzip -f -d t10k-images-idx3-ubyte.gz
!gzip -f -d t10k-labels-idx1-ubyte.gz
```
