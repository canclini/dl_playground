# Table of Contents
* [About](#user-content-about)
* [Prerequisites](#user-content-prerequisites)
* [CNN classifier](#user-content-run-a-cnn-classifier-on-mnist)
* [DCGAN with Keras and Tensorflow](#user-content-dcgan-with-keras-and-tensorflow)
* [Jupyter Notebooks on FloydHub](#user-content-gpu-powered-jupyter-notebooks)
* [Summary](#user-content-summary)

# About
Training Neural Networks can take a very long time on a standard CPU. It is therefore recommended to use GPU powered machines. This can be achieved by running your code on some virtual machines in [AWS](https://aws.amazon.com/ec2/) or by using some of the new services like [FloydHub](https://www.floydhub.com/) or [Valohai](https://valohai.com)

For development purpose, I'd like to use my local ecosystem and only want to use the GPU powered (and chargeable) services when doing the final training. Nevertheless, the code should be more or less the same locally and remote and the handling of using external services has to be very easy.

Bases on a [CNN classifier](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py) from [Aymeric Damien](https://github.com/aymericdamien)  and the more complex [DCGAN Implementation with Tensorflow and Keras](https://medium.com/towards-data-science/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0 ) from [Rowel Atienza](https://medium.com/@rowel) I try to give some ideas how to run the same code on the local machine and on [FloydHub](https://www.floydhub.com/) which has GPU powered docker containers.

## Prerequisites
### local installation
All the code here was only tested on a Mac running Mac OS X.
A local version of Python needs to be installed. [Anaconda](https://www.continuum.io/downloads) is a good choice.
As the many examples for MachineLearning and DeepLearning have different requirements (like python 2 or 3) I use [conda environments](https://conda.io/docs/using/envs.html) to isolate them for the projects.

There is an `environment.yml` file in this repository
to install the packages, run `conda env create -f environment.yml`. This should install the following packages with its dependencies.
```
python 3.5
numpy
matplotlib
keras 2.0.2
tensorflow 1.0.1
```

After all dependencies are installed the environment can be activated with `source activate dl_playground`

### FloydHub initialization
to run the code on a GPU powered system, without messing with virtual machines on AWS, I recently found FloydHub to be a nice alternative.

1. Get an account on [FloydHub](https://www.floydhub.com/). Currently it comes with 100 hours of free GPU usage.
2. Install the command line client `pip install -U floyd-cli`
3. Login with `floyd login`

Currently there is only a tensorflow-1.0 docker container with Keras 1.2.2. The used code examples are written for Keras 2.0.
The additional required package `keras==2.0.2` can be added to the [floyd_requirements.txt](floyd_requirements.txt) file. All the listed packages are installed in the docker container when the job gets executed.  

## run a CNN classifier on MNIST
The tensorflow based CNN classifier on MNIST from [Aymeric Damien](https://github.com/aymericdamien) is a good place to start with, as it is a clean and straightforward implementation. Also it takes about 5 minutes when executed locally on a standard CPU.

Get the [convolutional_network.py](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py) by either repo clone, raw download or  
```bash
wget https://raw.githubusercontent.com/aymericdamien/TensorFlow-Examples/master/examples/3_NeuralNetworks/convolutional_network.py
```

### run local
run the code
```bash
python convolutional_network.py
```
Depending on the machine it takes a few minutes.

### run on FloydHub
The idea is now to run the exact same code on a GPU powered system on FloydHub. The interaction is done via the `floyd` command line interface and/or via the web interface.

1. First initialize a new project on FloydHub by running 
```bash 
floyd init dl_playground
```
2. then submit the code to floyd with 
```bash
floyd run --gpu --env tensorflow-1.0 "python convolutional_network.py"
```

The code is the uploaded to FloydHub, a Docker container is started in which the code is executed with the power of some GPU.
The progress can be seen online or via the command line tool. See the [documentation](http://docs.floydhub.com/) for more information.

The first run takes a bit longer (~ 3 minutes) as it initiates the docker image (I think). The execution time for the CNN code is about 37s.

## DCGAN with Keras and Tensorflow
Now to something more complex with a longer execution time. 
[Rowel Atienza](https://medium.com/@rowel) wrote a nice [blog post explaining Generative Adversarial Networks (GAN)](https://medium.com/towards-data-science/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0) accompanied by a [Deep Convolution implementation](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py) with Tensorflow and Keras.

### code modifications
the original code does not run "as is" on FloydHub for 2 reasons:

**There is no Display**

generating plots with `matplotlib` does require a `$DISPLAY` to generate plots. To be used on a pure server environment the [matplotlib backend](https://matplotlib.org/faq/usage_faq.html#what-is-a-backend) has to be changed:
```python
import matplotlib
matplotlib.use('Agg')
```

**FloydHub has a predefined output directory**

All output on FloydHub has to be written into the predefined directory `/output`
Therefor a command line parameter --out-path` is added with a default value to `.`

Get the [implementation code](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py). 
```bash
wget https://raw.githubusercontent.com/roatienza/Deep-Learning-Experiments/master/Experiments/Tensorflow/GAN/dcgan_mnist.py
```

### run local
run the code
```bash
python convolutional_network.py
```
I ran the code with 100 training steps instead of 10000 and it took 32 minutes. So, it would take about **53 hours** with 10000 steps.

### run on FloydHub
initialize a new project if not already done and submit the code to floyd:
```bash
floyd init dl_playground
floyd run --env tensorflow-1.0 --gpu "python dcgan_mnist.py --out-path=/output"
```

runtime on FloydHub with 10000 training steps: ~2 hours

## GPU powered Jupyter Notebooks
Also worth to look into is the ability to run [jupyter notebooks on FloydHub](http://docs.floydhub.com/guides/jupyter/). 
For example you can run the [dl_course](https://tensorchiefs.github.io/dl_course/) notebooks on a GPU powered system:

```bash
git clone https://github.com/tensorchiefs/dl_course.git
cd dl_course
echo "keras==2.0.2" > floyd_requirements.txt
floyd init dl_course
floyd run --env tensorflow-1.0 --gpu --mode jupyter
```

Be careful. The Notebook Server has the be stopped manually, otherwise it can become expensive. For small models the notebook server can also be started on a standard CPU.

# Summary 
With small changes, existing code examples can be executed on GPU powered systems. When developing something new, the design adjustments are small to be able to run on FloydHub and the performance improvements are definitely worth it.
