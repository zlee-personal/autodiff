from torch.nn import Linear, ReLU, MSELoss, Module, ParameterList, Sigmoid
from torch import tensor
from torch.optim import SGD, Adam, RMSprop
import numpy as np
import torch
import matplotlib.pyplot as plt

class Model(Module):
    def __init__(self, sizes) -> None:
        super().__init__()
        self.layers = ParameterList()
        for size in sizes:
            self.layers.append(Linear(*size))
        self.activation = ReLU()
    
    def forward(self, x):
        out = self.layers[0](x)
        for layer in self.layers[1:]:
            out = layer(self.activation(out))
        return out

if __name__ == '__main__':

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)


    sizes = [(1, 500), (500, 500), (500, 1)]
    model = Model(sizes)
    criterion = MSELoss()
    optimizer = Adam(params=model.parameters())
    
    x = tensor(np.linspace(0, 2*np.pi, 100).reshape((100, 1)), dtype=torch.float32)
    y_target = torch.sin(x)
    
    loss_val = 1

    line_base, line1 = ax.plot(x.detach().numpy(), y_target.detach().numpy(), 'b-', x.detach().numpy(), model(x).detach().numpy(), 'r-')

    while loss_val > 0.0001:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_target)
        loss_val = loss.item()
        loss.backward()
        optimizer.step()
        print(loss_val)
        line1.set_ydata(model(x).detach().numpy())
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.show(block=True)
