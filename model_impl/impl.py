import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns


class NANetwork(nn.Module):
    def __init__(self, d_in_out, m_hidden, gamma=0.95):
        super().__init__()
        self.d = d_in_out
        self.m = m_hidden

        self.W = nn.Parameter(torch.randn(self.m, self.d))

        self.register_buffer('gamma', torch.tensor(gamma))

        self.register_buffer('b', torch.empty(self.m).uniform_(0, 2 * torch.pi))

        self.register_buffer('H', torch.zeros(self.d, self.m))
        self.register_buffer('g', torch.zeros(self.m))

    
    def _phi(self, x):
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return torch.cos(self.W @ x + self.b) 

    def forward(self, x, r):

        h = self._phi(x)

        if r == 1:
            p = (1 / self.m) * (self.g @ h)
            return (1 / p) * (self.H @ h) + x
        
        if r == 0:
            self.H.mul_(self.gamma).add_(torch.outer(x, h) / self.m)
            self.g.add_(h)
            return h

        return h
    
def visualize_H(model, title="Macierz Pamięci H"):
    h_matrix = model.H.detach().cpu().numpy()
    
    plt.figure(figsize=(12, 6))

    sns.heatmap(h_matrix, cmap='viridis', cbar=True)
    
    plt.title(title)
    plt.xlabel("Neurony ukryte (m_hidden)")
    plt.ylabel("Wymiary wejściowe (d_model)")
    plt.show()



def reset_model_state(model):
    model.H.zero_()
    model.g.zero_()
    
def run_experiment(model, embed, token_ids):
    reset_model_state(model)
    x_seq = embed(token_ids)
    for i in range(x_seq.size(0)):
        model.forward(x_seq[i], r=0)
    return model.H.detach().cpu().numpy().copy()

if __name__ == "__main__":
    vocab_size = 1000  
    d_model = 16       # 'x' (self.d) 
    m_hidden = 128     # 'h' (self.m) 
    
    embed = nn.Embedding(vocab_size, d_model)
    model = NANetwork(d_model, m_hidden)
    
    
    kontekst_a = torch.tensor([10, 20, 30]) 
    kontekst_b = torch.tensor([500, 510, 520])
    
    
    h_a = run_experiment(model, embed, kontekst_a)
    h_b = run_experiment(model, embed, kontekst_b)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(h_a, ax=ax1, cmap='magma', cbar=False)
    ax1.set_title("Memory H: Context 1")
    ax1.set_ylabel("Input Features (d)")
    ax1.set_xlabel("Hidden Neurons (m)")
    
    sns.heatmap(h_b, ax=ax2, cmap='magma', cbar=False)
    ax2.set_title("Memory H: Context 2")
    ax2.set_ylabel("Input Features (d)")
    ax2.set_xlabel("Hidden Neurons (m)")
    
    plt.tight_layout()
    plt.show()