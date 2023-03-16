import numpy as np

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

    
def vector_sum():
    pass


if __name__ == '__main__':
    x = np.random.uniform(0, 200, size=(10,))
    y = 3 * x + 5
    
    
    w = Var(0, 0)
    b = Var(0, 0)
    params = [w, b]

    learning_rate = 0.00006
    
    loss = Var(100000, 0)

    while loss.val > 0.0001:
        loss = None
        for n in range(len(x)):
            for i in range(len(params)):
                # we're only considering derivative with respect to 1 param at a time
                for param in params:
                    param.zero_grad()
                params[i].grad = 1
                
                loss = ((w * x[n] + b) - y[n])**2
                params[i].val -= loss.grad * learning_rate
        print(f'Loss: {loss.val}')
        print(f'w: {w.val}')
        print(f'b: {b.val}')

        
