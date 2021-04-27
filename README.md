
![title](https://user-images.githubusercontent.com/35846424/115941468-740aef00-a473-11eb-9440-7d086786de35.png)
<br/><br/>
Physics-aware training (PAT) is a method to train real physical systems with backpropagation. It was introduced in Wright, Logan G. & Onodera, Tatsuhiro *et al* (2021)<sup>[1](#how-to-cite-this-code)</sup> to train physical neural networks (PNNs) - neural networks whose building blocks are physical systems.

This repository is a PyTorch-based implementation of Physics-Aware Training. It lets users build Physical Neural Networks and automates many of the necessary steps to train them with Physics-Aware Training. To use an existing physical system as a building block in a neural network, users have to supply a class that receives batches of input data and processes them in the physical system. After specifying the trainable parameters, the system can be trained with this code. The methodology is demonstrated on an illustrative example of (nonlinear) coupled pendula. 

This repository also gives users access to documented reference code to implement or modify PAT.

## A graphical introduction to *Physical Neural Networks* and *Physics-Aware Training*
<br/><br/>
### Physical neural networks
![PNN](https://user-images.githubusercontent.com/35846424/115941478-7c632a00-a473-11eb-9f32-5167f55be062.png)
<br/><br/>
Physical neural networks are hierarchical computations whose building blocks are physical systems. The controllable conditions of a physical system are partitioned into inputs (red) and parameters (orange). By letting the system evolve in time, it naturally performs computations until outputs are read out (blue). Physical neural networks exploit these natural computations and enable multi-layered physical computations. Physics-aware training is the algorithm used to find the optimal control parameters.
<br/><br/>

### Physics-aware training
![PAT](https://user-images.githubusercontent.com/35846424/115941482-7e2ced80-a473-11eb-88b4-90b7fb784905.png)
<br/><br/>
Physics-aware training (PAT) is a gradient-descent algorithm that allows backpropagation through any physical system for which a digital model can be trained. As shown above, 1. inputs and parameters are sent into the physical system, which 2. propagate through the system. 3. The outputs of the system are compared to the intended outputs and 4. the gradient on the parameters is calculated by the differentiable digital model. 5. With this gradient, the paramters can be updated (Reprinted from Wright, L.G. & Onodera, T. *et al* [2021]). This repository implements the PAT algorithm and simplifies the training of differentiable digital models. Users only need to supply a class that controls the physical system.
<br/><br/>

### Users supply a class that executes an experiment
![define-exp](https://user-images.githubusercontent.com/35846424/115942312-9bfc5180-a477-11eb-9491-a4fe53fc8fa8.png)
<br/><br/>

Users need to supply a class that controls the physical system. When calling an instance of the class, the supplied code needs to initiate a physical computation with the given inputs and parameters.

### The ExpModule class learns a differentiable model of the experiment
![instantiate-ExpModule](https://user-images.githubusercontent.com/35846424/115963590-f9cb8080-a4ed-11eb-8bc8-92777de2be24.png)
<br/><br/>

The training of a differentiable digital model is facilitated by the ExpModule class. Users need to characterize the input and output dimensions of the physical system, just like, for example, a `torch.nn.Linear(d_in, d_out)`. Additionally, the trainable parameters and their experimentally allowable ranges need to be specified. Then the code can train a differentiable digital model for the physical system.

### A pnn.Module defines the physical neural network
![define-pnn](https://user-images.githubusercontent.com/35846424/115942317-a28ac900-a477-11eb-9f7c-e639c8557689.png)
<br/><br/>

Finally, users need to specify how the physical system is used in a multi-layer Physical Neural Network. These can be trained with a training loop just like any other PyTorch model

# Demonstrations

- [Coupled Pendula on a 2-dimensional dataset](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/master/Example-Pendula%20on%202D%20dataset.ipynb)
  An illustrative example of coupled pendula classifying simple distributions in a 2-dimensional plane, akin to https://playground.tensorflow.org/. The physical system is controlled by partitioning controllable initial conditions into inputs and parameters, to achieve >90% classification accuracy on multiple datasets.'
 
https://user-images.githubusercontent.com/35846424/115789949-ef956f00-a393-11eb-8814-cf4cb8ada98d.mp4
 
- [Coupled Pendula on vowel dataset](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/master/Example-Coupled%20Pendula%20on%20vowel%20dataset.ipynb) 
  This example shows the coupled pendula chain solving a slightly more complex task, that of [vowel classification](https://homepages.wmich.edu/~hillenbr/voweldata.html). Here, the       physical system is controlled by changing in-place parameters like the coupling constants and natural frequencies of the pendula to achieve 95% classification accuracy on a vowel classification datasets.
  
 ![download](https://user-images.githubusercontent.com/35846424/115791885-3c2e7980-a397-11eb-9a95-ef1804034fe9.png) 
  
- [Mechanical Multimode Oscillator on MNIST](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/master/Example-Speaker%20on%20MNIST.ipynb)
  A simulated replication of the mechanical multimode oscillator example classifying handwritten digits from the MNIST dataset as presented in Wright, Logan G. & Onodera, Tatsuhiro *et al* (2021)[^1].

# Getting started

A convenient starting point to understand the code in this repository is by running the example notebook of coupled pendula classifying points in a 2-dimensional cartesian plane. The notebook walks through all steps of creating and training a Physical Neural Network: 

1. Specifying a physical system and its trainable parameters,
2. Training a differentiable digital model that emulates the physical system,
3. Inserting the physical system into a neural network, and
4. Training the neural network with a physical forward- and digital backward-pass.

The notebook containing this example can be found at [Example-Pendula on 2D dataset.ipynb](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/master/Example-Pendula%20on%202D%20dataset.ipynb)

The backend code associated with each of those steps is in separate files:

[modules.py](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/master/modules.py) contains the class ExpModule in which users specify a physical system and its trainable parameters.

[digital.py](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/master/digitaltwin.py) contains the class DigitalTwin which is used to train a differentiable model that emulates the physical system.

[pnn.py](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/master/pnn.py) contains the pnn.Module class which is an extension of nn.Module and used to define the Physical Neural Network. It also contains code that modifies the training step of a neural network to respect the constraints of a physical system.

The backbone of Physics-Aware Training is a PyTorch Autograd Function that is defined in [lib/pat_utils.py](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/master/lib/pat_utils.py). It passes inputs through the user-supplied physical system on the forward-pass but through the digital model on the backward-pass.

## How you can make it work with your experiments

The goal of this repository is to set up Physical Neural Networks whose forward pass can consiste of any physical system:

```python
class PhysicalNeuralNetwork(pnn.Module):
    def __init__(self):
        super(pnn.Module, self).__init__()
        
        self.experiment = ExpModule(**experimentargs)     

    def forward(self, x):
       
        x = self.experiment(x)

        return output
```

In order to use a physical system in a fashion shown above, a few steps are required.
The first requirement is an `Experiment` class which processes batched input samples.
Specifically, the class will receive a list of inputs (in a `torch.tensor`) and needs to return a batch of processed outputs:

| Arguments                                              |     | Returns   |
| ---                                                    | --- | ---       |  
| (input 1, experimental parameters 1, hyperparameters)  | --> |  output 1 |
| (input 2, experimental parameters 2, hyperparameters)  | --> |  output 2 |
| (input 3, experimental parameters 3, hyperparameters)  | --> |  output 3 |
|  ...                                             |     |  ...      |

 Above is an abstract represenation of the mapping the experiment class needs to provide. 
 Varying inputs and experimental parameters need to be mapped to corresponding outputs. 
 The hyperparameters characterizing the experiment will be constant from run to run.

The following code are slightly adapted excerpts from the [exp.py](https://github.com/mcmahon-lab/PAT-demo-code/blob/master/exp.py) file.

To set up the experiment, define a custom class for it:

```python
class CoupledPendula():
    '''
    This class propagates inputs through a system of coupled pendula.
    Machine learning inputs will be encoded in the inital positions of the
    pendula, the position of the pendula at time T corresponds to the outputs.
    In this simulation, the natural frequency, damping, drive amplitude, drive phase,
    coupling, and intial velocity are in principle all trainable parameters.
    The time for how long pendula are being evolved is a constant hyperparameter.
    '''
```

Define a custom `__init___` method whose arguments are those parameters that stay constant in all experiment runs.
For a real experiment, connections to external devices could also be established at this point. Any additional arguments needed to set up
the experiment can be passed along through the `**kwargs` argument, for example MAC addresses of external devices.

```python
    def __init__(
            self,
            Tmax, # Simulation time after which final position is measured [s].
            **kwargs):
        
        # save hyperparameter as attribute to access it later
        self.Tmax = Tmax
        
        ...do additional preparations for the experiment...
```

Next, define a custom `__call__` function with the signature `(inputs, experimental parameter 1, experimental parameter 2, ...)`. The function needs to return `outputs` containing the processed samples.
Here, inputs are encoded in the `input_angles` the trainable parameters are `ω0, wd, ...`. The outputs are the final angles after evolving the pendula system for `Tmax`.
The `__call__` method will handle the following variables:

| `__call__` arguments          | type            | shape                                     | 
| ---                           | ---             | ---                                       | 
| inputs                        | torch.tensor    | \[batch size, input dimension\]           | 
| experimental parameters 1     | torch.tensor    | \[batch size, dimension of parameter 1\]  | 
| experimental parameters 2     | torch.tensor    | \[batch size, dimension of parameter 2\]  | 
| ...                           | ...             | ...                                       |   
| output                        | torch.tensor    | \[batch size, output dimension\]          |   

```python
    def __call__(
            self,      
            initial_angles, # initial angles encode inputs [rad]
            ω0, # eigenfrequency of each pendulum [s]
            ωd, # driving frequency of each pendulum [s]
            Ad, # driving amplitude of each pendulum [rad]
            v0, # inital velocity of each pendulum [rad/s]
            coupling, # universal coupling between pendula [rad/s^2 / rad]
            γ, # universal damping of pendula [(rad/s) / (rad/s) / s]
            encoding_amplitude, # multiplier for inputs
            phid, # driving phase
            **kwargs): 
        
        ...process inputs...
        
        return final_angles
```

To include the experiment in a Physical Neural Network, initiate an `ExpModule`. 
An `ExpModule` should be thought of as any trainable transformation like just like any PyTorch `nn.Module`, e.g. a fully connected layer.
The `ExpModule` needs more parameters during its initialization but eventually performs very similarly:

| `Exp Module` arguments:       | Meaning            | type                                    | 
| ---                           | ---                | ---                                     | 
| Exp | The name of the above defined experiment class.| class | 
| input_dim | Dimension of inputs to the ExpModule. | int | 
| output_dim | Dimension of outputs from the ExpModule. | int | 
| input_range | Lower and upper bound on allowable inputs to the experiment. | list of floats | 
| hparams | Values of hyperparameters with which `Exp.__init___` is called. | dict | 
| params_reg | Dimensions of trainable parameters with which `Exp.__call___` is called. | dict | 
| params_range | Lower and upper bound on allowable values for trainable parameters. | dict | 

This exemplary call shows how the above defined coupled pendula exeriment can be set up as an `ExpModule`.
`n_pendula` are the number of pendula in the experiment which determine the input and outputs dimension of the `ExpModule` and also the number of trainable parameters in many cases.

```python
pendulaargs = dict(
    Exp = CoupledPendula,
    input_dim = n_pendula, 
    output_dim = n_pendula, 
    input_range = [-0.5*np.pi,0.5*np.pi],
    hparams = dict(
        Tmax = 2.
    ),
    params_reg = dict(
       ω0 = [n_pendula],
       ωd = [n_pendula],
       Ad = [n_pendula],
       v0 = [n_pendula],
       coupling = [n_pendula],
       γ = [n_pendula],
       encoding_amplitude = [1],
       phid = [1]),
    params_range = {
       'ω0' : [5.,15.],
       'ωd' : [0.,0.],
       'Ad' : [0.,0.],
       'v0' : [0.,0.],
       'coupling' : [10, 30],
       'γ' : [0., 0.],
       'encoding_amplitude' : [1., 1.],
       'T' : [2,2],
       'phid' : [0,0]},
)  

pendula = ExpModule(**pendulaargs)
```

Inserting this code into a `pnn.Module`, we can define a Physical Neural Network. For more details, please see the [Coupled Pendula on a 2-dimensional dataset](https://github.com/mcmahon-lab/PAT-demo-code/blob/master/Example-Pendula%20on%202D%20dataset.ipynb) example.

```python
class PendulumNet(pnn.Module):
    def __init__(self):
        super(pnn.Module, self).__init__()
        
        self.pendula = ExpModule(**pendulaargs)     

    def forward(self, x):
       
        x = self.pendula(x)

        return output
```

# How to cite this code

If you use Physics-Aware Training in your research, please consider citing the following paper:

[//]: # (:radioactive: TODO -- insert paper reference)

Logan G. Wright, Tatsuhiro Onodera, Martin M. Stein, Tianyu Wang, Darren T. Schachter, Zoey Hu, and Peter L. McMahon (2021). Deep physical neural networks enabled by a backpropagation algorithm for arbitrary physical systems. *Manuscript in preparation.*

# License

The code in this repository is released under the following license:

[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

A copy of this license is given in this repository as license.txt.
