import torch
import torch.nn as nn
import clip
import numpy as np

class TextEncoder(nn.Module):
    def __init__(self, run_name, temperature):
        super().__init__()

        self.clip,_ = clip.load(run_name) 
        self.clip.requires_grad_(False) #freeze CLIP
        self.clip.eval()

        self.text_dim = self.clip.text_projection.shape[-1]
        self.logit_scale = nn.Parameter(torch.ones([])* np.log(1 / temperature))

    def forward(self,x): # x: text token
        x = self.clip.encode_text(x).float() #(N,text_dim)
        
        return x 
        

#class Adapter(nn.Module):
#    def __init__(self, text_dim, alpha=0.2):
#        super().__init__()
#        self.linear1 = nn.Sequential(
#                    nn.Linear(
#                        in_features=text_dim,
#                        out_features=text_dim//8
#                    ),
#                    nn.ReLU(),
#        )
#        self.linear2 = nn.Linear(
#                in_features=text_dim//8,
#                out_features=text_dim
#            )
#        
#        nn.init.zeros_(self.linear2.weight)
#        nn.init.zeros_(self.linear2.bias)
#            
#        self.dropout = nn.Dropout(0.1)
#        self.alpha = nn.Parameter(torch.tensor(alpha))
#
#    def forward(self,x):
#        res = x
#        x_down = self.dropout(self.linear1(x)) # compress to lower
#        x_up = self.linear2(x_down)
#        alpha = torch.sigmoid(self.alpha)
#        x = (1 - alpha) * res + alpha * x_up
#
#        return x



