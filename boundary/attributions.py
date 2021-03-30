try:
    import torch
    from captum.attr import IntegratedGradients
    from captum.attr import Saliency
    from captum.attr import DeepLift
    from captum.attr import Occlusion
    from captum.attr import NoiseTunnel
    print("Successfully import Pytorch and Captum")
except:
    print("Pytorch backend or Captum is not found.")

try:
    import tensorflow as tf
    if not tf.executing_eagerly():
        print(f"Eager execution is not enabled")

    if not tf.__version__.startswith('2'):
        print(
            f"The current version of TF ({tf.version.VERSION}) is not supported. Please insteall TF2 instead."
        )
    print("Successfully import Tensorflow")
except:
    print("Tensorflow backend is not found.")

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


def TF2_SaliencyGradient(model,
                         x,
                         y_onthot,
                         multiply_with_input=False,
                         **kwargs):
    
    if not isinstance(x, tf.Tensor):
        x = tf.Variable(tf.constant(x))
    with tf.GradientTape() as tape:
        tape.watch(x)
        qoi = model(x, training=False) * y_onthot
        qoi = tf.reduce_sum(qoi, axis=-1)
    attr = tape.gradient(qoi, x)
    if multiply_with_input:
        attr *= x
    return attr


def TF2_SmoothGradient(model,
                       x,
                       y_onthot,
                       stdevs=0.1,
                       n_steps=50,
                       multiply_with_input=False,
                       **kwargs):
    if not isinstance(x, tf.Tensor):
        x = tf.Variable(tf.constant(x))
    grad = tf.zeros_like(x)
    for _ in range(1, n_steps + 1):
        x_in = x + tf.random.normal(x.shape, stddev=stdevs)
        with tf.GradientTape() as tape:
            tape.watch(x_in)
            qoi = model(x_in, training=False) * y_onthot,
            qoi = tf.reduce_sum(qoi, axis=-1)

        grad += tape.gradient(qoi, x_in) / n_steps
    if multiply_with_input:
        attr = grad * x
    else:
        attr = grad
    return attr


def TF2_IntegratedGradient(model,
                           x,
                           y_onthot,
                           n_steps=50,
                           baseline=None,
                           **kwargs):
    if not isinstance(x, tf.Tensor):
        x = tf.Variable(tf.constant(x))
        
    if baseline is None:
        baseline = tf.zeros_like(x)
    elif isinstance(baseline, float) or isinstance(baseline, int):
        baseline = tf.zeros_like(x) + baseline
    
    if not isinstance(baseline, tf.Tensor):
        baseline = tf.constant(baseline)

    assert baseline.shape == x.shape

    grad = tf.zeros_like(x)

    for i in range(n_steps + 1):
        x_in = baseline + (x - baseline) * i / n_steps
        with tf.GradientTape() as tape:
            tape.watch(x_in)
            qoi = model(x_in, training=False) * y_onthot
            qoi = tf.reduce_sum(qoi, axis=-1)
        input_grad = tape.gradient(qoi, x_in)
        grad += input_grad / n_steps

    attr = grad * (x - baseline)
    return attr
