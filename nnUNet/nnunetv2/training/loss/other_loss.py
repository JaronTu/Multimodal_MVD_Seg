import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss.ntx_ent_loss import NTXentLoss


def cc(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

def cc_3D(img1, img2):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(torch.sum(img1 **
                                                                      2, dim=-1)) * torch.sqrt(torch.sum(img2**2, dim=-1)))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()

# # cscada directly project to B*(c*w*h*d) from cscada_net.py
# self.f1 = nn.Sequential(nn.Conv2d(filters[4], 64, kernel_size=3, padding=1, bias=True),
#                         nn.Conv2d(64, 16, kernel_size=1))
# self.g1 = nn.Sequential(nn.Linear(in_features=4096, out_features=1024),
#                         nn.ReLU(),
#                         nn.Linear(in_features=1024, out_features=256))
similar_criterion = nn.CosineSimilarity()
# projection before contrastive module
def contrast_loss():
    pos_s2t_similar = similar_criterion(high_r_s_sb, high_r_t_tb) / train_params['temp_fac']
    den_s2t1_similar = similar_criterion(high_r_s_sb, high_r_s_tb) / train_params['temp_fac']
    den_s2t2_similar = similar_criterion(high_r_s_sb, high_r_t_sb) / train_params['temp_fac']
    contrast_loss_s2t = -torch.log(torch.exp(pos_s2t_similar) / (torch.exp(pos_s2t_similar) +
                                                                 torch.sum(torch.exp(den_s2t1_similar) + torch.exp(
                                                                     den_s2t2_similar))))
    return contrast_loss


def distill_kl(self, y_s, y_t, T=1):
    '''
    vanilla distillation loss with KL divergence
    '''
    if y_s.shape[1] == 1:
        y_s = torch.cat([y_s,torch.zeros_like(y_s)],1)
        y_t = torch.cat([y_t,torch.zeros_like(y_t)],1)

    p_s = F.log_softmax(y_s/T+1e-40, dim=1)
    p_t = F.softmax(y_t/T, dim=1)

    loss = F.kl_div((p_s), p_t, reduction='mean') * (T**2)

    return loss


def l2_loss(input, target, channel_wise=False, T=1):  # input/target: [b,dim=8,...]
    if channel_wise == True:
        # bs,dim,d,w,h = input.shape
        # input = torch.flatten(input,2) # torch.reshape(input,(bs,dim,-1))
        # target = torch.flatten(target,2) # torch.reshape(target,(bs,dim,-1))

        # loss = F.kl_div(F.log_softmax(input/T, dim=2), F.softmax(target/T, dim=2), reduction='mean') * (T**2)

        loss = F.kl_div(F.log_softmax(input / T, dim=1), F.softmax(target / T, dim=1), reduction='mean') * (T ** 2)
        return loss
    else:
        return torch.mean(torch.abs(input - target).pow(2))