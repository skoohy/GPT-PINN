import torch
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class P(nn.Module):
    def __init__(self, w1, w2, w3, w4, w5, b1, b2, b3, b4, b5):
        super().__init__()
        self.layers = [2, 20, 20, 20, 20, 1]
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        
        self.linears[0].weight.data = torch.Tensor(w1)
        self.linears[1].weight.data = torch.Tensor(w2)
        self.linears[2].weight.data = torch.Tensor(w3)
        self.linears[3].weight.data = torch.Tensor(w4)
        self.linears[4].weight.data = torch.Tensor(w5).view(1,self.layers[4])
        
        self.linears[0].bias.data = torch.Tensor(b1)
        self.linears[1].bias.data = torch.Tensor(b2)
        self.linears[2].bias.data = torch.Tensor(b3)
        self.linears[3].bias.data = torch.Tensor(b4)
        self.linears[4].bias.data = torch.Tensor(b5).view(-1)
        
        self.activation = nn.Tanh()
        
    def forward(self, x):       
        a = x
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)        
        a = self.linears[-1](a)
        return a