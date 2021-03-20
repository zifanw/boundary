import torch
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
import numpy as np


def PT_SaliencyGradient(model,
                        x,
                        y_onthot,
                        multiply_with_input=False,
                        device='cuda:0',
                        **kwargs):
    input = torch.tensor(x).to(device)
    model = model.to(device)
    model.eval()
    saliency = Saliency(model)
    target = torch.tensor(np.argmax(y_onthot, -1)).to(device)
    attribution_map = saliency.attribute(input, target=target, abs=False)
    if multiply_with_input:
        attribution_map *= input
    return attribution_map.detach().cpu().numpy()


def PT_SmoothGradient(model,
                      x,
                      y_onthot,
                      multiply_with_input=False,
                      device='cuda:0',
                      n_steps=50,
                      stdevs=0.15,
                      **kwargs):
    input = torch.tensor(x).to(device)
    model = model.to(device)
    model.eval()
    saliency = NoiseTunnel(Saliency(model))
    target = torch.tensor(np.argmax(y_onthot, -1)).to(device)
    attribution_map = saliency.attribute(input,
                                         n_samples=n_steps,
                                         target=target,
                                         stdevs=stdevs,
                                         abs=False)

    if multiply_with_input:
        attribution_map *= input
    return attribution_map.detach().cpu().numpy()


def PT_IntegratedGradient(model,
                          x,
                          y_onthot,
                          baseline=None,
                          n_steps=50,
                          method='riemann_trapezoid',
                          device='cuda:0',
                          **kwargs):
    input = torch.tensor(x, requires_grad=True).to(device)
    model = model.to(device)
    model.eval()
    saliency = IntegratedGradients(model)
    target = torch.tensor(np.argmax(y_onthot, -1)).to(device)
    if baseline is not None:
        baseline = torch.tensor(baseline).to(device)

    # if method == Riemann.trapezoid:
    #     return list(np.linspace(0, 1, n))
    # elif method == Riemann.left:
    #     return list(np.linspace(0, 1 - 1 / n, n))
    # elif method == Riemann.middle:
    #     return list(np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n))
    # elif method == Riemann.right:
    #     return list(np.linspace(1 / n, 1, n))
    #  Ref: https://github.com/pytorch/captum/blob/master/captum/attr/_utils/approximation_methods.py

    attribution_map = saliency.attribute(input,
                                         target=target,
                                         baselines=baseline,
                                         n_steps=n_steps,
                                         method=method)
    return attribution_map.detach().cpu().numpy()


def PT_DeepLIFT(model, x, y_onthot, baseline=None, device='cuda:0', **kwargs):
    input = torch.tensor(x, requires_grad=True).to(device)
    model = model.to(device)
    model.eval()
    saliency = DeepLift(model)
    target = torch.tensor(np.argmax(y_onthot, -1)).to(device)
    if baseline is not None:
        baseline = torch.tensor(baseline).to(device)
    attribution_map = saliency.attribute(input,
                                         target=target,
                                         baselines=baseline)
    return attribution_map.detach().cpu().numpy()