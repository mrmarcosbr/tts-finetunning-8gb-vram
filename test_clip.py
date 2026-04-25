import torch
x = torch.tensor([10.0], requires_grad=True)
y = x * x
y.backward()
print("Grad before:", x.grad)
torch.nn.utils.clip_grad_norm_([x], max_norm=0.0)
print("Grad after:", x.grad)
