"""
Modified from https://github.com/pytorch/vision/blob/master/torchvision/models/ResNetEvoNorm.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (
            sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i))
        )


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


@torch.jit.script
def instance_std(x, eps):
    var = torch.var(x, dim=(2, 3), keepdim=True).expand_as(x)
    if torch.isnan(var).any():
        var = torch.zeros(var.shape)
    return torch.sqrt(var + eps)


@torch.jit.script
def group_std(x, eps):
    N, C, H, W = x.size()
    groups = 32
    groups = C if groups > C else groups
    x = x.view(N, groups, C // groups, H, W)
    var = torch.var(x, dim=(2, 3, 4), keepdim=True).expand_as(x)
    return torch.sqrt(var.add(eps)).view(N, C, H, W)


class EvoNorm2D(nn.Module):
    def __init__(
        self,
        input,
        non_linear=True,
        version="S0",
        efficient=True,
        affine=True,
        momentum=0.9,
        eps=1e-5,
        groups=32,
        training=True,
    ):
        super(EvoNorm2D, self).__init__()
        self.non_linear = non_linear
        self.version = version
        self.training = training
        self.momentum = momentum
        self.efficient = efficient
        if self.version == "S0":
            self.swish = MemoryEfficientSwish()
        self.groups = groups
        self.eps = nn.Parameter(torch.FloatTensor([eps]), requires_grad=False)
        if self.version not in ["B0", "S0"]:
            raise ValueError("Invalid EvoNorm version")
        self.insize = input
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, self.insize, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, self.insize, 1, 1))
            if self.non_linear and (
                (self.version == "S0" and not self.efficient) or self.version == "B0"
            ):
                self.v = nn.Parameter(torch.ones(1, self.insize, 1, 1))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
            self.register_buffer("v", None)
        self.register_buffer("running_var", torch.ones(1, self.insize, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_var.fill_(1)

    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(x.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        if self.version == "S0":
            if self.non_linear:
                if not self.efficient:
                    num = x * torch.sigmoid(
                        self.v * x
                    )  # Original Swish Implementation, however memory intensive.
                else:
                    num = self.swish(
                        x
                    )  # Experimental Memory Efficient Variant of Swish
                return num / group_std(x, eps=self.eps) * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
        if self.version == "B0":
            if self.training:
                var = torch.var(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
                self.running_var.mul_(self.momentum)
                self.running_var.add_((1 - self.momentum) * var)
            else:
                var = self.running_var

            if self.non_linear:
                den = torch.max(
                    (var + self.eps).sqrt(), self.v * x + instance_std(x, eps=self.eps)
                )
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta