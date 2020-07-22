import torch
from utility import perturbations_4
user = torch.tensor([1, 0, 4, 0, 0, 0, 2])
#res = perturbations_3(user)
res = perturbations_4(user, 10)
print(res)