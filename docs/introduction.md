## A short introduction to *Physical Neural Networks* and *Physics-Aware Training*
<br/><br/>
### Physical neural networks
![g5888](https://user-images.githubusercontent.com/35846424/116467819-1ba26b80-a83e-11eb-8042-20d746c4f1b2.png)
<br/><br/>
Physical neural networks are hierarchical computations whose building blocks are physical systems. The controllable conditions of a physical system are partitioned into inputs (red) and parameters (orange). By letting the system evolve in time, it naturally performs computations until outputs are read out (blue). Physical neural networks exploit these natural computations and enable multi-layered physical computations. Physics-aware training is the algorithm used to find the optimal control parameters.
<br/><br/>

### Physics-aware training
![g6030](https://user-images.githubusercontent.com/35846424/116467836-2230e300-a83e-11eb-8f97-ce6c9003b89e.png)
<br/><br/>
Physics-aware training (PAT) is a gradient-descent algorithm that allows backpropagation through any physical system for which a digital model can be trained. As shown above, 1. inputs and parameters are sent into the physical system, which 2. propagate through the system. 3. The outputs of the system are compared to the intended outputs and 4. the gradient on the parameters is calculated by the differentiable digital model. 5. With this gradient, the paramters can be updated. This repository implements the PAT algorithm and simplifies the training of differentiable digital models.

For details on PAT, please refer to the Supplementary Section 1 of our paper. It lays out the intuition for why PAT works and explains its mathematical formulation: https://doi.org/10.1038/s41586-021-04223-6
<br/><br/>

[Back to the repository](https://github.com/mcmahon-lab/Physics-Aware-Training)
