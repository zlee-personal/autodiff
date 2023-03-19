from abc import ABC, abstractclassmethod
import numpy as np
from typing import List, Union
import matplotlib.pyplot as plt

class Var:
    """
    A variable that represents a scalar or vector value. It automatically keeps track of a computation graph by storing the operator and list of operands that created it.

    Attributes
    ----------
    val : np.ndarray
        The scalar or vector value.
    children : List[Var]
        A list of the variable's children.
    op : Op, optional
        The operation that produced the variable, if any.
    """
    def __init__(self, val, children: List['Var'] = None, op: 'Op' = None):
        self.val = val
        self.children = children if children else []
        self.op = op
        self.grad = np.zeros_like(val)
        
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(op={self.op.__name__}, val={self.val}, grad={self.grad})'
        
    def backwards(self, grad) -> None:
        self.grad = self.grad + grad
        if self.op is None:
            assert not self.children
        else:
            self.op.handle_children(self.children, self.grad)

    def __add__(self, other: 'Var') -> 'Var':
        return Add.forward(self, other)

    def __mul__(self, other: 'Var') -> 'Var':
        return Mul.forward(self, other)

    def __matmul__(self, other: 'Var') -> 'Var':
        return MatMul.forward(self, other)

    def __pow__(self, other: 'Var') -> 'Var':
        return Pow.forward(self, other)

    def __neg__(self) -> 'Var':
        return Neg.forward(self)

    def __sub__(self, other: 'Var') -> 'Var':
        return Sub.forward(self, other)

        
class Op(ABC):
    """A base class for defining operations on variables and their corresponding vector Jacobian products.
    """
    def __repr__(self):
        return f'{self.__class__.__name__}()'

    @abstractclassmethod
    def forward(*args: 'Var'):
        """Takes in `Var`s and produces a new Var
        """
        pass

    @abstractclassmethod
    def handle_children(children: List['Var'], grad):
        """Calculate the VJP with incoming gradient. Assuming current op "f", operands (`children`) x_1, ... x_n,
        and incoming `grad` dL/df, calculate and send the correct gradients to each child dL/dx_1, ..., dL/dx_2 though
        that child's `backwards` method.

        Parameters
        ----------
        children : List[Var]
            List of children or operands for the operation.
        grad : numerical
            Incoming gradient dL/df, where f is the operation result.
        """
        pass


class Add(Op):
    @staticmethod
    def forward(left, right):
        return Var(left.val + right.val, [left, right], __class__)

    @staticmethod
    def handle_children(children, grad):
        children[0].backwards(grad)
        children[1].backwards(grad)


class Mul(Op):
    @staticmethod
    def forward(left, right):
        return Var(left.val * right.val, [left, right], __class__) 

    @staticmethod
    def handle_children(children, grad):
        children[0].backwards(grad * children[1].val)
        children[1].backwards(grad * children[0].val)

class Sub(Op):
    @staticmethod
    def forward(left: 'Var', right: 'Var') -> 'Var':
        return Var(left.val - right.val, [left, right], __class__)

    @staticmethod
    def handle_children(children: List['Var'], grad: Union[float, np.ndarray]) -> None:
        children[0].backwards(grad)
        children[1].backwards(-grad)

class Pow(Op):
    @staticmethod
    def forward(left: 'Var', right: 'Var') -> 'Var':
        return Var(left.val ** right.val, [left, right], __class__)

    @staticmethod
    def handle_children(children: List['Var'], grad: Union[float, np.ndarray]) -> None:
        left, right = children
        left.backwards(grad * right.val * left.val ** (right.val - 1))
        right.backwards(grad * left.val ** right.val * np.log(left.val))

class Neg(Op):
    @staticmethod
    def forward(x: 'Var') -> 'Var':
        return Var(-x.val, [x], __class__)

    @staticmethod
    def handle_children(children: List['Var'], grad: Union[float, np.ndarray]) -> None:
        children[0].backwards(-grad)

class MatMul(Op):
    @staticmethod
    def forward(left: 'Var', right: 'Var') -> 'Var':
        return Var(np.dot(left.val, right.val), [left, right], __class__)

    @staticmethod
    def handle_children(children: List['Var'], grad: np.ndarray) -> None:
        left, right = children
        # Calculate the gradients with respect to left and right
        left_grad = np.dot(grad, right.val.T)
        right_grad = np.dot(left.val.T, grad)

        # Perform the backward pass
        left.backwards(left_grad)
        right.backwards(right_grad)

class ReLU(Op):
    @staticmethod
    def forward(x: 'Var') -> 'Var':
        return Var(np.maximum(0, x.val), [x], __class__)

    @staticmethod
    def handle_children(children: List['Var'], grad: np.ndarray) -> None:
        children[0].backwards(grad * (children[0].val > 0))

class Sum(Op):
    @staticmethod
    def forward(x: 'Var') -> 'Var':
        return Var(np.sum(x.val), [x], __class__)

    @staticmethod
    def handle_children(children: List['Var'], grad: float) -> None:
        children[0].backwards(np.ones_like(children[0].val) * grad)
        

def initialize_adam_parameters(params):
    adam_params = []
    for param in params:
        adam_params.append({
            'm': np.zeros_like(param.val),
            'v': np.zeros_like(param.val),
            't': 0
        })
    return adam_params

def update_parameters_adam(params, grads, adam_params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    for param, grad, adam_param in zip(params, grads, adam_params):
        adam_param['t'] += 1
        adam_param['m'] = beta1 * adam_param['m'] + (1 - beta1) * grad
        adam_param['v'] = beta2 * adam_param['v'] + (1 - beta2) * (grad ** 2)

        m_hat = adam_param['m'] / (1 - beta1 ** adam_param['t'])
        v_hat = adam_param['v'] / (1 - beta2 ** adam_param['t'])

        param.val -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)


def compute_dC_dA(A, B):
    M, N = A.shape
    P = B.shape[1]
    
    # Create an identity matrix of shape (M, M)
    I = np.eye(M)
    
    # Reshape the identity matrix and B to make them broadcast-compatible
    I_expanded = I[:, np.newaxis, :, np.newaxis]
    B_expanded = B.T[np.newaxis, :, np.newaxis, :]
    
    # Compute dC/dA using broadcasting
    dC_dA = B_expanded * I_expanded
    
    return dC_dA

def kronecker_delta(i, j):
    return 1 if i == j else 0

def compute_dC_dA2(A, B):
    M, N = A.shape
    P = B.shape[1]
    
    dC_dA = np.zeros((M, P, M, N))
    
    for i in range(M):
        for j in range(P):
            for k in range(M):
                for l in range(N):
                    dC_dA[i, j, k, l] = kronecker_delta(k, i) * B[l, j]
                    
    return dC_dA


if __name__ == '__main__':

    A = np.arange(6).reshape((3,2))
    B = np.arange(8).reshape((2,4))

    dL_dC = np.arange(3*4).reshape((3,4))
    
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)


    # create neural network
    hidden_sizes = [(1, 500), (500, 500), (500,1)]

    weights = []
    biases = []
    for size in hidden_sizes:
        weights.append(Var(np.random.uniform(-1/np.sqrt(size[1]), 1/np.sqrt(size[1]), size=size)))
        biases.append(Var(np.random.uniform(-1/np.sqrt(size[1]), 1/np.sqrt(size[1]), size=(1, size[1]))))

    def forward(x, w, b):
        '''forward pass for neural network'''
        batch_size = x.val.shape[0]
        r = x @ w[0] + Var(np.ones((batch_size, 1))) @ b[0]
        for i in range(1, len(w)):
            r = ReLU.forward(r) @ w[i] + Var(np.ones((batch_size, 1))) @ b[i]
        return r


    # Define input data and target
    x = Var(np.linspace(-np.pi, np.pi, 100).reshape((100, 1)))
    y_target = Var(np.sin(x.val)+2*(np.sin(3*x.val)))
    
    
    line_base, line1 = ax.plot(x.val, y_target.val, 'b-', x.val, forward(x, weights, biases).val, 'r-')

    loss_val = 1



    # Initialize Adam optimizer state variables
    adam_params = initialize_adam_parameters(weights + biases)

    loss_val = float('inf')
    while loss_val > 0.0001:
        # zero gradients
        for param in weights + biases:
            param.grad = np.zeros_like(param.grad)

        # Forward pass
        y_pred = forward(x, weights, biases)

        # Calculate loss (mean squared error)
        loss = Sum.forward((y_pred - y_target) ** Var(2.0)) * Var(1 / x.val.shape[0])

        # Backward pass
        loss.backwards(1)

        # Update weights and biases using Adam optimizer
        update_parameters_adam(weights + biases, [param.grad for param in weights + biases], adam_params)

        loss_val = loss.val

        print("Loss:", loss_val)
        line1.set_ydata(forward(x, weights, biases).val)
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    plt.show(block=True)
