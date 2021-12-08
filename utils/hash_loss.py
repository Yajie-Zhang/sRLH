import torch.nn as nn
import torch



class Hash_Loss(nn.Module):
    def __init__(self, code_length, gamma):
        super(Hash_Loss, self).__init__()
        self.code_length = code_length
        self.gamma = gamma

    def forward(self, F, C, S, device):
        hash_loss = ((self.code_length*S - F @ C.t()) ** 2).sum()
        quantization_loss = ((torch.abs(F)-torch.ones(F.shape).to(device))**2).sum()
        loss = (hash_loss + self.gamma * quantization_loss) / (F.shape[0] * C.shape[0])
        return loss