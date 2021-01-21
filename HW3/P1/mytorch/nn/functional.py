import numpy as np
from mytorch import tensor
from mytorch.autograd_engine import Function
import math


def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(a.data.T, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return (tensor.Tensor(grad_output.data.T),)


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor: {}".format(type(a).__name__))
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.log(a.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return (tensor.Tensor(grad_output.data / a.data),)


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(np.exp(a.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return (tensor.Tensor(grad_output.data * np.exp(a.data)),)


"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(
                type(a).__name__, type(b).__name__))

        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(a.data + b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(
                type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad

        c = tensor.Tensor(a.data - b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = np.ones(a.shape) * grad_output.data

        grad_b = -np.ones(b.shape) * grad_output.data
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data.sum(axis=axis, keepdims=keepdims),
                          requires_grad=requires_grad, is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None


# TODO: Implement more Functions below


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(
                type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad

        c = tensor.Tensor(a.data*b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = b.data * grad_output.data

        grad_b = a.data * grad_output.data
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor') or \
                a.data.shape != b.data.shape or 0 in b.data:
            raise Exception("Both args must be Tensors: {}, {}".format(
                type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad

        c = tensor.Tensor(a.data/b.data, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = 1/b.data * grad_output.data

        grad_b = (-a.data)/(np.square(b.data)) * grad_output.data
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors: {}, {}".format(
                type(a).__name__, type(b).__name__))
        ctx.save_for_backward(a, b)
        requires_grad = a.requires_grad or b.requires_grad

        c = tensor.Tensor(np.einsum('ik, kj -> ij', a.data, b.data), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = np.einsum('ik, kj -> ij', grad_output.data, b.data.T)
        grad_b = np.einsum('ik, kj -> ij', a.data.T, grad_output.data)
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class ReLU(Function):
    @staticmethod
    def forward(ctx, z):
        if not (type(z).__name__ == 'Tensor'):
            raise Exception("Arg must be a Tensor: {}".format(
                type(z).__name__))
        ctx.save_for_backward(z)
        requires_grad = z.requires_grad

        c = tensor.Tensor(np.where(z.data > 0, z.data, 0), requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        z = ctx.saved_tensors
        z = z[0]
        grad_z = np.where(z.data > 0, 1, 0) * grad_output.data
        grad_z = tensor.Tensor(unbroadcast(grad_z, z.shape))
        return (grad_z,)


def log_sum_x_trick(predicted):

    a = np.amax(predicted.data, axis=1)
    a = a.reshape((1, len(a)))
    a = tensor.Tensor(np.tile(a.T, (1, predicted.data.shape[-1])))
    return a


def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape

    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.

    # Tip 2: Remember to divide the loss by batch_size; this is equivalent
    #        to reduction='mean' in PyTorch's nn.CrossEntropyLoss

    e_x = predicted.exp()
    log_e_x = e_x.log()
    a = log_sum_x_trick(predicted)
    x_n_offset = predicted - a

    exp_xn_offset = x_n_offset.exp()

    sum_exp_xn_offset = exp_xn_offset.sum(axis=1, keepdims=True)
    log_sum_exp_xn_offset = sum_exp_xn_offset.log()
    denominator = a + log_sum_exp_xn_offset
    log_softmax = log_e_x - denominator

    labels = to_one_hot(target, num_classes)
    prod = log_softmax*labels
    total = prod.sum()
    batch_size = tensor.Tensor(-batch_size)

    total = total / batch_size

    return total


def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]

    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad=True)


class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.

        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.

        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution

        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        if not (type(x).__name__ == 'Tensor' and type(weight).__name__ == 'Tensor' and type(bias).__name__ == 'Tensor'):
            raise Exception("All args must be Tensors: {},{}, {}".format(
                type(x).__name__, type(weight).__name__), type(bias).__name__)
        # TODO: Save relevant variables for backward pass
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)
        ctx.output_size = output_size
        requires_grad = x.requires_grad or weight.requires_grad or bias.requires_grad
        # TODO: Initialize output with correct size
        out = np.zeros((batch_size, out_channel, output_size))

        for i in range(batch_size):
            for j in range(out_channel):
                curr = 0
                for k in range(0, input_size-kernel_size+1, stride):
                    out[i][j][curr] = np.sum(x.data[i, :, k:k+kernel_size]
                                             * weight.data[j]) + bias.data[j]
                    curr += 1

        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.
        out = tensor.Tensor(out, requires_grad=requires_grad,
                            is_leaf=not requires_grad)

        # TODO: Put output into tensor with correct settings and return
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Finish Conv1d backward pass. It's surprisingly similar to the forward pass.
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        output_size = ctx.output_size
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape
        flip_w = np.flip(weight.data, axis=2)
        grad_output_dw = np.zeros(
            (batch_size, out_channel, get_conv1d_output_size(input_size, kernel_size, 1)))
        grad_x = np.zeros(x.shape)
        for i in range(batch_size):
            for j in range(out_channel):
                for k in range(output_size):
                    grad_output_dw[i][j][k*stride] = grad_output.data[i][j][k]

        grad_output_dx = np.pad(grad_output_dw, ((0, 0), (0, 0),
                                                 (kernel_size-1, kernel_size-1)),
                                mode='constant', constant_values=0)

        for i in range(batch_size):
            for j in range(out_channel):
                for k in range(in_channel):
                    for l in range(input_size):
                        grad_x[i][k][l] += np.sum(grad_output_dx[i][j]
                                                  [l:l+kernel_size] * flip_w[j][k])

        grad_w = np.zeros(weight.shape)
        for i in range(batch_size):
            for j in range(out_channel):
                for k in range(in_channel):
                    for l in range(kernel_size):
                        grad_w[j][k][l] += np.sum(grad_output_dw[i][j] *
                                                  x.data[i][k][l:l+input_size-kernel_size+1])

        grad_b = np.sum(grad_output.data, axis=(0, 2))

        grad_b = tensor.Tensor(grad_b, requires_grad=True)
        grad_w = tensor.Tensor(grad_w, requires_grad=True)
        grad_x = tensor.Tensor(grad_x, requires_grad=True)
        return grad_x, grad_w, grad_b


def get_conv1d_output_size(input_size, kernel_size, stride):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.

        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """
    return ((input_size - kernel_size)//stride) + 1

class Pow(Function):
    @staticmethod
    def forward(ctx, a, b):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Exp must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        ctx.exponent = b
        requires_grad = a.requires_grad
        c = tensor.Tensor(a.data**b, requires_grad=requires_grad,
                          is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        b = ctx.exponent
        return (tensor.Tensor(grad_output.data * b * a.data**(b-1)), None)



class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensor.Tensor(b_data, requires_grad=a.requires_grad)
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1-b)
        return tensor.Tensor(grad),


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1-out**2)
        return tensor.Tensor(grad),


class Slice(Function):
    @staticmethod
    def forward(ctx,x,indices):
        '''
        Args:
            x (tensor): Tensor object that we need to slice
            indices (int,list,Slice): This is the key passed to the __getitem__ function of the Tensor object when it is sliced using [ ] notation.
        '''
        requires_grad = x.requires_grad
        ctx.indices = indices
        ctx.original_shape = x.shape
        
        result = tensor.Tensor(x.data[indices], requires_grad=requires_grad, is_leaf=not requires_grad)
        return result


    @staticmethod
    def backward(ctx,grad_output):
        indices = ctx.indices
        original_shape = ctx.original_shape
        result = np.zeros(original_shape)
        result[indices] = grad_output.data 

        return tensor.Tensor(result), None
class Cat(Function):
    @staticmethod
    def forward(ctx,*args):
        '''
        Args:
            args (list): [*seq, dim] 
        
        NOTE: seq (list of tensors) contains the tensors that we wish to concatenate while dim (int) is the dimension along which we want to concatenate 
        '''
        *seq, dim = args


        grad = False
        ctx.data =  []
        ctx.dim = dim
        result = None
        for t in seq:
            if t.requires_grad:
                grad = True
            
            if result is None: #First
                result = t.data
            else:
                assert(result is not None)
                result = np.concatenate((result, t.data), axis=dim)
            ctx.data.append(t.data)
            

        result = tensor.Tensor(result, requires_grad=grad)
        result.is_leaf = not result.requires_grad
        return result


    @staticmethod
    def backward(ctx,grad_output):
        dim = ctx.dim 
        data = ctx.data 

        split_points = []
        for i in range(0, len(data)-1):
            if len(split_points) == 0:
                split_points.append(data[i].shape[dim])
            else:
                
                split_points.append(data[i].shape[dim] + split_points[-1])
        

            
        grad = np.split(grad_output.data, split_points, axis=dim)
        for i in range(len(grad)):
            grad[i] = tensor.Tensor(grad[i])
        
        grad.append(None)
        grad = tuple(grad)
        return grad
            



