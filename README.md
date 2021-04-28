
![title](https://user-images.githubusercontent.com/35846424/115941468-740aef00-a473-11eb-9440-7d086786de35.png)
<br/><br/>
Physics-aware training (PAT) is a method to train real physical systems with backpropagation. It was introduced in Wright, Logan G. & Onodera, Tatsuhiro *et al* (2021)<sup>[1](#how-to-cite-this-code)</sup> to train physical neural networks (PNNs) - neural networks whose building blocks are physical systems.

This repository is a PyTorch-based implementation of Physics-Aware Training. It lets users build Physical Neural Networks and automates many of the necessary steps to train them with Physics-Aware Training. To use an existing physical system as a building block in a neural network, users have to supply a class that receives batches of input data and processes them in the physical system. After specifying the trainable parameters, the system can be trained with this code. The methodology is demonstrated on an illustrative example of (nonlinear) coupled pendula. 

This repository also gives users access to documented reference code to implement or modify PAT.

# How to get started

- To learn about Physical Neural Networks, Physics-Aware Training, and the scope of this repository, have a look at the [Introduction](https://github.com/mcmahon-lab/Physics-Aware-Training/tree/main/docs/introduction.md).
- To see pedagogical examples with simulations of real experiments, head over to the [Demonstrations](https://github.com/mcmahon-lab/Physics-Aware-Training/tree/main/examples/README.md).
- To apply PAT to new experiments, see [How to make it work with your experiments](https://github.com/mcmahon-lab/Physics-Aware-Training/tree/main/docs/how_to_create_own_exp.md).
- To see the source code of `physics_aware_training`, see the package in the [physics_aware_training](https://github.com/mcmahon-lab/Physics-Aware-Training/tree/main/physics_aware_training) folder.


# How to cite this code

If you use Physics-Aware Training in your research, please consider citing the following paper:

>Logan G. Wright, Tatsuhiro Onodera, Martin M. Stein, Tianyu Wang, Darren T. Schachter, Zoey Hu, and Peter L. McMahon (2021). Deep physical neural networks enabled by a backpropagation algorithm for arbitrary physical systems. *Manuscript in preparation.*

# License

The code in this repository is released under the following license:

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

A copy of this license is given in this repository as [license.txt](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/main/license.txt).
