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
Physics-aware training (PAT) is a gradient-descent algorithm that allows backpropagation through any physical system for which a digital model can be trained. As shown above, 1. inputs and parameters are sent into the physical system, which 2. propagate through the system. 3. The outputs of the system are compared to the intended outputs and 4. the gradient on the parameters is calculated by the differentiable digital model. 5. With this gradient, the paramters can be updated (Reprinted from Wright, L.G. & Onodera, T. *et al* [2021]). This repository implements the PAT algorithm and simplifies the training of differentiable digital models.
<br/><br/>

### Users supply a class that executes an experiment
![g5110](https://user-images.githubusercontent.com/35846424/116467963-4bea0a00-a83e-11eb-99e5-2b804d1f2525.png)
<br/><br/>

Users need to supply a class that controls the physical system. When calling an instance of the class, the supplied code needs to initiate a physical computation with the given inputs and parameters and measure the outputs of the physical system.

### The ExpModule class learns a differentiable model of the experiment
![g4851](https://user-images.githubusercontent.com/35846424/116467920-3d035780-a83e-11eb-8015-31f195e03162.png)
<br/><br/>

The training of a differentiable digital model is facilitated by the `ExpModule` class. Users need to characterize the input and output dimensions of the physical system, just like, for example, a `torch.nn.Linear(d_in, d_out)`. Additionally, the trainable parameters and their experimentally allowable ranges need to be specified. Then the code can train a differentiable digital model for the physical system.

### A pnn.Module defines the physical neural network
![g5005](https://user-images.githubusercontent.com/35846424/116468008-586e6280-a83e-11eb-9245-506bedfcd317.png)
<br/><br/>

Finally, users need to specify how the physical system is used in a multi-layer Physical Neural Network. PNNs can then be trained with a training loop just like any other PyTorch model. 

An example notebook walking through all the steps shown here in more detail is available in [Example 1-Coupled Pendula on 2D dataset](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/main/examples/Example%201-Coupled%20Pendula%20on%202D%20dataset.ipynb).

[Back to the repository](https://github.com/mcmahon-lab/Physics-Aware-Training)
