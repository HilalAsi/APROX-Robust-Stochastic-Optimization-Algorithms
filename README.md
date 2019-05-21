# APOX-Robust-Stochastic-Optimization-Algorithms

TensorFlow and Pytorch open source implementation for the aProx optimization methods from the paper:

[*The importance of better models in stochastic optimization*](https://arxiv.org/abs/1903.08619)

by Hilal Asi and John Duchi.

---

This repository provides implementation for the aProx optimizatin algorithms (Truncated and Truncated-Adagrad), which improve the robustness of classical optimization algorithms (e.g. SGD and Adagrad) to the stepsize value. The folders Optimizers_tf and Optimizers_pytorch include the implementation for TensorFlow and Pytorch, respectively. Examples of using these optimizers can be found in the files example_tf.py and example_pytorch.py.


## Contact

***Code author:*** Hilal Asi

***Pull requests and issues:*** @HilalAsi

## Citation

If you use this code, please cite our paper:
```
@article{asi2019importance,
  title={The importance of better models in stochastic optimization},
  author={Asi, Hilal and Duchi, John C},
  journal={arXiv preprint arXiv:1903.08619},
  year={2019}
}
```
