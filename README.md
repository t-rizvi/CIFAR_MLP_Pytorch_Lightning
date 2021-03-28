# CIFAR_MLP_Pytorch_Lightning

A Multilayer Perceptron (MLP) Neural Network is trained using Pytorch LIghtning library.
CIFAR dataset is used to trian the neural network.

Different Experiments are performed and results are observed. Type of experiments and validation accuracy of network is given below:


Version 1:	B_SIze: 32  H_Layers: 1	  H Neurons: 512  Optim: SGD  Sigmoid   Val_Acc: 0.4706

Version 2:	B_SIze: 32  H_Layers: 1	  H Neurons:1512  Optim: SGD  Sigmoid   Val_Acc: 0.4626

Version 3:	B_SIze: 32  H_Layers: 1	  H Neurons:1512  Optim: SGD  RELU      Val_Acc: 0.5089

Version 4:	B_SIze: 32  H_Layers: 1	  H Neurons:1512  Optim: ADAM RELU      Val_Acc: 0.5114

Version 5:	B_SIze: 32  H_Layers: 1	  H Neurons:1512  Optim: ADAM RELU      Val_Acc: 0.5389   Augmentation

Version 6:	B_SIze: 32  H_Layers: 2	  H Neurons:1512  Optim: ADAM RELU      Val_Acc: 0.5389   Augmentation
