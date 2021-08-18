import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class reweight_loss(nn.Module):
    def __init__(self):
        super(reweight_loss, self).__init__()
        
    def forward(self, out, T, target):
        loss = 0.
        out_softmax = F.softmax(out, dim=1)
        for i in range(len(target)):
            temp_softmax = out_softmax[i]
            temp = out[i]
            temp = torch.unsqueeze(temp, 0)
            temp_softmax = torch.unsqueeze(temp_softmax, 0)
            temp_target = target[i]
            temp_target = torch.unsqueeze(temp_target, 0)
            pro1 = temp_softmax[:, target[i]] 
            out_T = torch.matmul(T.t(), temp_softmax.t())
            out_T = out_T.t()
            pro2 = out_T[:, target[i]] 
            beta = pro1 / pro2
            beta = Variable(beta, requires_grad=True)
            cross_loss = F.cross_entropy(temp, temp_target)
            _loss = beta * cross_loss
            loss += _loss
        return loss / len(target)


class reweighting_revision_loss(nn.Module):
    def __init__(self):
        super(reweighting_revision_loss, self).__init__()
        
    def forward(self, out, T, correction, target):
        loss = 0.
        out_softmax = F.softmax(out, dim=1)
        for i in range(len(target)):
            temp_softmax = out_softmax[i]
            temp = out[i]
            temp = torch.unsqueeze(temp, 0)
            temp_softmax = torch.unsqueeze(temp_softmax, 0)
            temp_target = target[i]
            temp_target = torch.unsqueeze(temp_target, 0)
            pro1 = temp_softmax[:, target[i]]
            T = T + correction
            T_result = T
            out_T = torch.matmul(T_result.t(), temp_softmax.t())
            out_T = out_T.t()
            pro2 = out_T[:, target[i]]    
            beta = (pro1 / pro2)
            cross_loss = F.cross_entropy(temp, temp_target)
            _loss = beta * cross_loss
            loss += _loss
        return loss / len(target)
    
def get_noisy_prob(transition_mat, clean_prob):
    return torch.matmul(transition_mat.T, clean_prob.unsqueeze(-1)).squeeze()

class reweight_loss_v2(nn.Module):
    def __init__(self):
        super(reweight_loss_v2, self).__init__()
        
    def forward(self, out, T, target):
        out_softmax = F.softmax(out, dim=1)
        noisy_prob = get_noisy_prob(T, out_softmax)
        pro1 = torch.gather(out_softmax, dim=-1, index=target.unsqueeze(1)).squeeze()
        pro2 = torch.gather(noisy_prob, dim=-1, index=target.unsqueeze(1)).squeeze()
        beta = pro1 / pro2
        beta = Variable(beta, requires_grad=True)
        cross_loss = F.cross_entropy(out, target, reduction='none')
        _loss = beta * cross_loss
        return torch.mean(_loss)

class reweighting_revision_loss_v2(nn.Module):
    def __init__(self):
        super(reweighting_revision_loss_v2, self).__init__()
        
    def forward(self, out, T, correction, target):
        out_softmax = F.softmax(out, dim=1)
        T = T + correction
        noisy_prob = get_noisy_prob(T, out_softmax)
        pro1 = torch.gather(out_softmax, dim=-1, index=target.unsqueeze(1)).squeeze()
        pro2 = torch.gather(noisy_prob, dim=-1, index=target.unsqueeze(1)).squeeze()
        beta = pro1 / pro2
        beta = Variable(beta, requires_grad=True)
        cross_loss = F.cross_entropy(out, target, reduction='none')
        _loss = beta * cross_loss
        return torch.mean(_loss)
