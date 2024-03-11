import torch

# input:[H,W,D]
# output:[C,H,W,D]
x = torch.zeros([7,12,24,24])
shp_x = x.shape
print(shp_x)
y = torch.zeros([1,12,24,24])
gt = y.long()
y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
y_onehot.scatter_(1, gt, 1)
print(y_onehot.shape)