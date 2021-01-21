import numpy as np
from mytorch import tensor
from mytorch.tensor import Tensor
from mytorch.nn.module import Module
from mytorch.nn.activations import Tanh, ReLU, Sigmoid
from mytorch.nn.util import PackedSequence, pack_sequence, unpack_sequence


class RNNUnit(Module):
    '''
    This class defines a single RNN Unit block.

    Args:
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the RNNUnit at each timestep
        nonlinearity (string): Non-linearity to be applied the result of matrix operations 
    '''

    def __init__(self, input_size, hidden_size, nonlinearity = 'tanh' ):
        
        super(RNNUnit,self).__init__()
        
        # Initializing parameters
        self.weight_ih = Tensor(np.random.randn(hidden_size,input_size), requires_grad=True, is_parameter=True)
        self.bias_ih   = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)
        self.weight_hh = Tensor(np.random.randn(hidden_size,hidden_size), requires_grad=True, is_parameter=True)
        self.bias_hh   = Tensor(np.zeros(hidden_size), requires_grad=True, is_parameter=True)

        self.hidden_size = hidden_size
        
        # Setting the Activation Unit
        if nonlinearity == 'tanh':
            self.act = Tanh()
        elif nonlinearity == 'relu':
            self.act = ReLU()

    def __call__(self, input, hidden = None):
        return self.forward(input,hidden)

    def forward(self, input, hidden = None):
        '''
        Args:
            input (Tensor): (effective_batch_size,input_size)
            hidden (Tensor,None): (effective_batch_size,hidden_size)
        Return:
            Tensor: (effective_batch_size,hidden_size)
        '''
        
        w_x = input@self.weight_ih.T() + self.bias_ih
        if hidden==None:
            data = np.zeros((input.shape[0], self.weight_hh.shape[0]))
            hidden = Tensor(data)

        w_h = hidden@self.weight_hh.T() + self.bias_hh
        res = w_x + w_h
        return self.act.forward(res)
        # raise NotImplementedError('Implement Forward')


class TimeIterator(Module):
    '''
    For a given input this class iterates through time by processing the entire
    seqeunce of timesteps. Can be thought to represent a single layer for a 
    given basic unit which is applied at each time step.
    
    Args:
        basic_unit (Class): RNNUnit or GRUUnit. This class is used to instantiate the unit that will be used to process the inputs
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the RNNUnit at each timestep
        nonlinearity (string): Non-linearity to be applied the result of matrix operations 

    '''

    def __init__(self, basic_unit, input_size, hidden_size, nonlinearity = 'tanh' ):
        super(TimeIterator,self).__init__()

        # basic_unit can either be RNNUnit or GRUUnit
        self.unit = basic_unit(input_size,hidden_size,nonlinearity)    

    def __call__(self, input, hidden = None):
        return self.forward(input,hidden)
    
    def forward(self,input,hidden = None):
        
        '''
        NOTE: Please get a good grasp on util.PackedSequence before attempting this.

        Args:
            input (PackedSequence): input.data is tensor of shape ( total number of timesteps (sum) across all samples in the batch, input_size)
            hidden (Tensor, None): (batch_size, hidden_size)
        Returns:
            PackedSequence: ( total number of timesteps (sum) across all samples in the batch, hidden_size)
            Tensor (batch_size,hidden_size): This is the hidden generated by the last time step for each sample joined together. Samples are ordered in descending order based on number of timesteps. This is a slight deviation from PyTorch.
        '''

        # Resolve the PackedSequence into its components
        data, sorted_indices, batch_sizes = input
        final_hidden_states= []
        prev_hidden = None
        output_pack = []
        start_point = 0
        for i in range(len(batch_sizes)):
            
            if prev_hidden is not None: hidden_state = self.unit.forward(data[start_point:start_point + batch_sizes[i]], prev_hidden[:batch_sizes[i]])
            else:
                assert(prev_hidden == None)
                hidden_state = self.unit.forward(data[:start_point  + batch_sizes[i]], None)
            output_pack.append(hidden_state)
            prev_hidden = hidden_state
            if i < len(batch_sizes) - 1 and batch_sizes[i+1] < batch_sizes[i]:
                for j in range(-1, batch_sizes[i+1] - batch_sizes[i] - 1, -1):
                    final_hidden_states.insert(0, hidden_state[j].unsqueeze(0))
            start_point += batch_sizes[i]
        
        for i in range(len(hidden_state)): # Add final values
            final_hidden_states.insert(0, hidden_state[i].unsqueeze(0))
        final_hidden_states = tensor.cat(final_hidden_states, 0)
        
        output_pack = tensor.cat(output_pack, 0)
        
        output_pack = PackedSequence(output_pack, sorted_indices, batch_sizes)
        return output_pack, final_hidden_states


class RNN(TimeIterator):
    '''
    Child class for TimeIterator which appropriately initializes the parent class to construct an RNN.
    Args:
        input_size (int): # features in each timestep
        hidden_size (int): # features generated by the RNNUnit at each timestep
        nonlinearity (string): Non-linearity to be applied the result of matrix operations 
    '''

    def __init__(self, input_size, hidden_size, nonlinearity = 'tanh' ):
        super().__init__(RNNUnit, input_size, hidden_size,nonlinearity)
