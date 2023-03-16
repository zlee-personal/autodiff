import numpy as np
import matplotlib.pyplot as plt

class Var:
    def __init__(self, val, grad):
        self.grad = grad
        self.val = val
        
    def __repr__(self):
        return f'{self.__class__}(val={self.val}, grad={self.grad})'

    def zero_grad(self):
        self.grad = 0 * self.grad

    def __add__(self, other):
        if isinstance(other, Var):
            return Var(self.val + other.val, self.grad + other.grad)
        else:
            return Var(self.val + other, self.grad)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        if isinstance(other, Var):
            return Var(self.val * other.val, self.grad * other.val + self.val * other.grad)
        else:
            return Var(self.val * other, self.grad * other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        return self.__add__(-1 * other)
    
    def __rsub__(self, other):
        return (-1 * self).__add__(other)
    
    def __pow__(self, other):
        other_val = None
        other_grad = None
        if isinstance(other, Var):
            other_val = other.val
            other_grad = other.grad
        else:
            other_val = other
            other_grad = 0 * other
            
        return Var(self.val ** other_val, self.val ** (other_val - 1) * (self.val * other_grad * np.emath.log(self.val) + other_val * self.grad))

    def __rpow__(self, other):
        assert not isinstance(other, Var)
        return Var(other ** self.val, other ** (self.val - 1) * (other * self.grad * np.emath.log(other)))
    
    def __matmul__(self, other):
        assert not isinstance(other, Var) or not np.any(other.grad)
        if isinstance(other, Var):
            other = other.val
        return Var(self.val @ other, self.grad @ other)

    
def vector_sum(vec):
    return Var(np.sum(vec.val), np.sum(vec.grad))


def element_to_var(e):
    return Var(e, 0)

def relu(x):
    grad = 0 if x.val < 0 else x.grad
    return Var(np.maximum(x.val, 0), grad)

relu_vec = np.vectorize(relu)

def tanh(x):
    return Var(np.tanh(x.val), 1 - np.tanh(x.grad)**2)

tanh_vec = np.vectorize(tanh)

get_val_vec = np.vectorize(lambda x: x.val)


np_to_var = np.vectorize(element_to_var)


if __name__ == '__main__':
    x = np.linspace(0, 5, 50).reshape(1, 50)
    y = np.sin(x)
    
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    hidden_sizes = [(10, 1), (10, 10), (5, 10), (1, 5)]

    weights = []
    biases = []
    for size in hidden_sizes:
        weights.append(np_to_var(np.random.normal(size=size)))
        biases.append(np_to_var(np.random.normal(size=(size[0], 1))))


    def forward(x, w, b):
        r = w[0] @ x + b[0]
        for i in range(1, len(w)):
            r = w[i] @ tanh_vec(r) + b[i]
        return r
        
    
    params = []
    for l in weights + biases:
        params += l.flatten().tolist()

    line_base, line1 = ax.plot(x.flatten(), y.flatten(), 'b-', x.flatten(), get_val_vec(forward(x, weights, biases)).flatten(), 'r-')

    learning_rate = 0.005
    
    loss = Var(100000, 0)

    while loss.val > 0.0001:
        loss = None
        for i in range(len(params)):
            # we're only considering derivative with respect to 1 param at a time
            for param in params:
                param.zero_grad()
            params[i].grad = 1
            
            loss = np.sum((forward(x, weights, biases) - y)**2)
            params[i].val -= loss.grad * learning_rate
        print(f'Loss: {loss.val}')
        line1.set_ydata(get_val_vec(forward(x, weights, biases)).flatten())
        fig.canvas.draw()
        fig.canvas.flush_events()

        
