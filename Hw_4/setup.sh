echo "### INSTALLING DEPENDENCIES ###"
pip install -r requirements.txt
echo "### DEPENDENCIES INSTALLED ###"

echo "### DOWNLOADING DATA ###"
mkdir data
cd data
echo "### DOWNLOADING CIFAR-10 ###"
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
echo "### DOWNLOADING CIFAR-100 ###"
wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvf cifar-100-python.tar.gz
rm cifar-100-python.tar.gz
echo "### DOWNLOADING MNIST ###"
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gzip -d train-images-idx3-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
gzip -d t10k-images-idx3-ubyte.gz
gzip -d t10k-labels-idx1-ubyte.gz
echo "### DATA DOWNLOADED ###"