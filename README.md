![g5382](https://user-images.githubusercontent.com/35846424/116468174-894e9780-a83e-11eb-97a8-88063e1653c0.png)
<br/><br/>
*Physics-Aware Training* (PAT) is a method to train real physical systems with backpropagation. It was introduced in Wright, Logan G. & Onodera, Tatsuhiro *et al.* (2022)<sup>[1](#how-to-cite-this-code)</sup> to train *Physical Neural Networks* (PNNs) - neural networks whose building blocks are physical systems.

In this repository, we use examples based on simulated nonlinear coupled oscillators, to show how PNNs can be constructed and trained using PAT in PyTorch. Instead of a conventional python package, most of the code in this repository resides within self-contained Jupyter notebook examples. We have deliberately taken this approach, in the hopes that it will allow users to more easily understand and adapt this code for their own use. In our paper, we have taken essentially the same approach and demonstrated the methodology on real experiments.

# Getting started

- To learn about *Physical Neural Networks*, *Physics-Aware Training*, and the scope of this repository, have a look at the [Introduction](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/main/docs/introduction.md).
- To see the examples that show how PNNs can be constructed and trained using PAT, see [Examples](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/main/docs/examples.md).

# How to cite this code

If you use *Physics-Aware Training* in your research, please consider citing the following paper:

> Wright, L.G., Onodera, T., Stein, M.M. et al. Deep physical neural networks trained with backpropagation. _Nature_ **601**, 549–555 (2022). https://doi.org/10.1038/s41586-021-04223-6

# License

The code in this repository is released under the following license:

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

A copy of this license is given in this repository as [license.txt](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/main/license.txt).
