
# How to make it work with your experiments

In order to use a physical system in as a neural network building block, these steps are required.
The first requirement is an `Experiment` class which processes batched input samples.
Specifically, the class will receive a list of inputs (in a `torch.tensor`) and needs to return a batch of processed outputs:

| Arguments                                              |     | Returns   |
| ---                                                    | --- | ---       |  
| (input 1, experimental parameters 1, hyperparameters)  | --> |  output 1 |
| (input 2, experimental parameters 2, hyperparameters)  | --> |  output 2 |
| (input 3, experimental parameters 3, hyperparameters)  | --> |  output 3 |
|  ...                                             |     |  ...      |

 Above is an abstract representation of the mapping the experiment class needs to provide. 
 Varying inputs and experimental parameters need to be mapped to corresponding outputs. 
 The hyperparameters characterizing the experiment will be constant from run to run.

The following snippets are slightly adapted excerpts from the [exp.py](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/master/examples/exp.py) file, showcasing a simulation of an example experiment class.

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

Inserting this code into a `pnn.Module`, we can define a Physical Neural Network. The full implementation of the `CoupledPendula` experiment class is located in [examples/exp.py](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/main/examples/exp.py), for more details on how to build a PNN with this experiment, please see the [Coupled Pendula on a 2-dimensional dataset](https://github.com/mcmahon-lab/Physics-Aware-Training/blob/main/examples/Example%201-Coupled%20Pendula%20on%202D%20dataset.ipynb) example.

```python
class PendulumNet(pnn.Module):
    def __init__(self):
        super(pnn.Module, self).__init__()
        
        self.pendula = ExpModule(**pendulaargs)     

    def forward(self, x):
       
        x = self.pendula(x)

        return output
```

[Back to the repository](https://github.com/mcmahon-lab/Physics-Aware-Training)
