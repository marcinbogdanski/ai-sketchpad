<h1> AI Sketchpad </h1>

Implementations of various Deep Learning architectures. Includes MLPs, CNNs, RNNs, Seq2Seq, GANs.

<h2 id="numpy"> Numpy Implementations </h2>

Neural networks implemented from scratch in numpy.

* Multi-Layer Perceptrons:
  * [Fully-Connected 1-Layer](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/NumpyNN/0010_FC_1Layer.ipynb) - classification on College Admissions dataset
  * [Fully-Connected 2-Layer for Classification](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/NumpyNN/0020_FC_2LayerClass.ipynb) - classification on Fashion MNIST
  * [Fully-Connected 2-Layer for Regression](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/NumpyNN/0020_FC_2Layer_Reg.ipynb) - regression on Bike Sharing dataset
* Recurrent Models:
  * [Vanilla RNN Char-Level Many-2-One](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/NumpyNN/1010_Char_RNN_Unfolded.ipynb) - synthetic counting task
  * [Vanilla RNN Char-Level Many-2-Many](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/NumpyNN/1020_Char_RNN_Dinosaurs.ipynb) - generate dinosaur names


<h2 id="numpy"> Keras Implementations </h2>

Neural networks implemented in `keras.layers` API

* Multi-Layer Perceptrons:
  * [Multi-Layer Perceptron](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/KerasNN/0100_MLP_College.ipynb) - classification on College Admissions
  * [Multi-Layer Perceptron](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/KerasNN/0120_MLP_MNIST.ipynb) - classification on MNIST
  * [Multi-Layer Perceptron](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/KerasNN/0150_MLP_IMBD.ipynb) - sentiment analysis on IMBD
* Convolutional Models:
  * [Convolutional Neural Network](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/KerasNN/1100_CNN_CIFAR10.ipynb) - classification on CIFAR-10
  * [CNN with Batch Normalization](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/KerasNN/1200_CNN_BN_CIFAR10.ipynb) - classification on CIFAR-10
  * [CNN with Data Augumentation](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/KerasNN/1250_Aug_CIFAR10.ipynb) - classification on CIFAR-10
  * [ResNet-50 in Keras Layers API](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/KerasNN/1300_ResNet50_Scratch.ipynb) - classification on Oxford VGG Flower 17 dataset
  * [ResNet-50 Transfer Learning](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/KerasNN/1400_ResNet50_Transfer.ipynb) - classification on Oxford VGG Flower 17 dataset
* Recurrent Models:
  * [Seq-2-Seq LSTM with Embeddings](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/KerasNN/2310_Seq2Seq_EngFr.ipynb) - English to French translation on a small corpus
* Generative Models
  * [Autoencoder](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/KerasNN/3010_AE_MNIST.ipynb) - fully connected autoencoder applied to MNIST
  * [Vanilla GAN](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/KerasNN/3110_GAN_FC_MNIST.ipynb) - fully connected GAN on MNIST dataset
  * [DCGAN](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/KerasNN/3210_DCGAN.ipynb) - deep convolutional GAN on CelebA dataset (incomplete)


<h2 id="datasets"> Datasets </h2>

This section included dataset preprocessing notebooks. These need to be run first before corresponding neural network notebooks.

* Image datasets
  * [Tiny ImageNet](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/Datasets/1300_TinyImageNet.ipynb) - download, explore and convert to .npz
  * [Oxford VGG Flowers 17](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/Datasets/1410_Oxford_VGG_Flowers.ipynb) - download, explore and convert to .npz
  * [Stanford Dogs](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/Datasets/1430_Stanford_Dogs.ipynb) - download, explore and convert to .npz

<h2 id="debugging"> Debugging Techniques </h2>

Debugging techniques. Track input/output distributions, individual neuron weights, gradients, preactivation histograms, 

**Note** this notebook has cool graphs, but no description

* [MLP with Sigmoid Activation](https://github.com/marcinbogdanski/ai-sketchpad/blob/master/DebugNN/Debug_Bikes.ipynb)
