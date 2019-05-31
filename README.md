# APROX: Robust Stochastic Optimization Algorithms

TensorFlow and Pytorch open source implementation for the aProx optimization methods from the paper:

[*The importance of better models in stochastic optimization*](https://arxiv.org/abs/1903.08619)

by [Hilal Asi](http://web.stanford.edu/~asi/) and [John Duchi](http://web.stanford.edu/~jduchi/).

---

This repository provides implementation for the aProx optimizatin algorithms (Truncated and Truncated-Adagrad), which improve the robustness of classical optimization algorithms (e.g. SGD and Adagrad) to the stepsize value. The folders Optimizers_tf and Optimizers_pytorch include the implementation for TensorFlow and Pytorch, respectively. Examples of using these optimizers can be found in the files example_tf.py and example_pytorch.py.

The following plots (from the paper) show the time-to-convergence as a function of the stepsize for various methods for CIFAR10 and Stanfrod-dogs datasets.

![CIFAR10 plot](https://github.com/HilalAsi/APOX-Robust-Stochastic-Optimization-Algorithms/blob/master/paper-plots/CIFAR10-plot.PNG "CIFAR10")

![Stanford dogs plot](https://github.com/HilalAsi/APOX-Robust-Stochastic-Optimization-Algorithms/blob/master/paper-plots/Stanford-dogs-plot.PNG "Stanford dogs")

## Contact

***Code author:*** Hilal Asi

***Pull requests and issues:*** @HilalAsi

## Citation

If you use this code, please cite our paper:
```
@article{asi2019importance,
  title={The importance of better models in stochastic optimization},
  author={Asi, Hilal and Duchi, John C},
  journal={arXiv:1903.08619 [math.OC]},
  year={2019}
}
```
