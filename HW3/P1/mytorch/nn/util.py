from mytorch import tensor
import numpy as np

class PackedSequence:
    
    '''
    Encapsulates a list of tensors in a packed seequence form which can
    be input to RNN and GRU when working with variable length samples
    
    ATTENTION: The "argument batch_size" in this function should not be confused with the number of samples in the batch for which the PackedSequence is being constructed. PLEASE read the description carefully to avoid confusion. The choice of naming convention is to align it to what you will find in PyTorch. 

    Args:
        data (Tensor):( total number of timesteps (sum) across all samples in the batch, # features ) 
        sorted_indices (ndarray): (number of samples in the batch for which PackedSequence is being constructed,) - Contains indices in descending order based on number of timesteps in each sample
        batch_sizes (ndarray): (Max number of timesteps amongst all the sample in the batch,) - ith element of this ndarray represents no.of samples which have timesteps > i
    '''
    def __init__(self,data,sorted_indices,batch_sizes):
        
        # Packed Tensor
        self.data = data # Actual tensor data

        # Contains indices in descending order based on no.of timesteps in each sample
        self.sorted_indices = sorted_indices # Sorted Indices
        
        # batch_size[i] = no.of samples which have timesteps > i
        self.batch_sizes = batch_sizes # Batch sizes
    
    def __iter__(self):
        yield from [self.data,self.sorted_indices,self.batch_sizes]
    
    def __str__(self,):
        return 'PackedSequece(data=tensor({}),sorted_indices={},batch_sizes={})'.format(str(self.data),str(self.sorted_indices),str(self.batch_sizes))


def pack_sequence(sequence): 
    '''
    Constructs a packed sequence from an input sequence of tensors.
    By default assumes enforce_sorted ( compared to PyTorch ) is False
    i.e the length of tensors in the sequence need not be sorted (desc).

    Args:
        sequence (list of Tensor): ith tensor in the list is of shape (Ti,K) where Ti is the number of time steps in sample i and K is the # features
    Returns:
        PackedSequence: data attribute of the result is of shape ( total number of timesteps (sum) across all samples in the batch, # features )
    '''
    
    lengths = set()
    indexes = []
    
    for i in range(len(sequence)): lengths.add(len(sequence[i]))
    
    
    ordered_lengths = sorted(lengths)
    ordered_lengths.reverse()
    sorted_tensors = []
    for length in ordered_lengths:
        for i in range(len(sequence)):
            if len(sequence[i]) == length:
                sorted_tensors.append(sequence[i])
                if i not in indexes: indexes.append(i)

    
    array_t = []
    batch_size = []
    

    

    for j in range(ordered_lengths[0]):
        sum = 0
        for i in range(len(sequence)):
            
            if len(sorted_tensors[i]) > j:
                
                array_t.append(sorted_tensors[i][j].unsqueeze(0))
                sum += 1
            else: break
            
        batch_size.append(sum)
    batch_size = np.array(batch_size)
    indexes = np.array(indexes)
    array_t = tensor.cat(array_t, dim=0)
    result = PackedSequence(array_t, indexes, batch_size)
  
    return result


def unpack_sequence(ps):
    '''
    Given a PackedSequence, this unpacks this into the original list of tensors.
    
    NOTE: Attempt this only after you have completed pack_sequence and understand how it works.

    Args:
        ps (PackedSequence)
    Returns:
        list of Tensors
    '''
    
    data = ps.data
    sorted_indices = ps.sorted_indices
    batch_sizes = ps.batch_sizes
    arr_tensors = [[] for _ in range(len(sorted_indices))]
    i = 0
    count = 0
    while i < len(data):
        for j in range(batch_sizes[count]):
            arr_tensors[j].append(data[i+j].unsqueeze(0))

        i += batch_sizes[count]
        count += 1
    for i in range(len(arr_tensors)): arr_tensors[i] = tensor.cat(arr_tensors[i], dim=0)

    result = [None for i in range(len(sorted_indices))]
    for i in range(len(sorted_indices)): result[i] = arr_tensors[list(sorted_indices).index(i)]
    return result

