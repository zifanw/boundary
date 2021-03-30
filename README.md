# Boundary Attribution
Implementation of [Boundary Attributions for Normal (Vector) Explanations](https://arxiv.org/pdf/2103.11257.pdf)

![image](media/demo.png)

## Setup
Run `sh install_lib.sh` to install the dependencies. 

Note that the results presented in the paper are using Pytorch models, but we also provide implementations for Tensorflow 2. If you are using Tensorflow, you must ensure you use eager execution (which is enabled by default in TF2).  The current code is tested using 
```
Python = 3.8.5
Pytorch = 1.6.0
Torchvision = 0.7.0

or 

Tensorflow = 2.4.0
```

## Supported Methods

| Methods        | Non-Boundary          | Boundary  |
| ------------- |:-------------:| -----:|
| Saliency Map     | Pytorch/TF | Pytorch/TF |
| Integrated Gradient      | Pytorch/TF      |   Pytorch/TF |
| Smooth Gradient | Pytorch/TF      |   Pytorch/TF |
| DeppLIFT | Pytorch      |    Pytorch |


## Compute Boundary Attributions (Pytorch)

A complete example is shown in `example_pytorch.ipynb`. The following code block should work with your own Pytorch model.

```python
import boundary
from boundary import BA_pytorch

# Load your pytorch/tf.keras model
model = load_model(...) 

# Load your data into numpy arrays
numpy_data_x, labels_onehot = load_your_data(...)

# Create attribution object 
big = BA_pytorch('BIG', use_boundary=True)

# Load the default parameters. You can also modify these parameters if needed. 
# These default parameters should work with one Titan RTX 3080.
parameters = boundary.PARAMETERS
pipeline = boundary.PIPELINE


# Compute attribution scores
attr = big.attribute(model,
                     numpy_data_x,
                     labels_onehot,
                     pipline=pipeline,
                     return_dis=False,
                     **parameters)
```

## Compute Boundary Attributions (Tensorflow 2)

A complete example is shown in `example_tensorflow.ipynb`. The following code block should work with your own `tf.keras` model.

```python
import boundary
from boundary import BA_tensorflow

# Load your pytorch/tf.keras model
model = load_model(...) 

# Load your data into numpy arrays
numpy_data_x, labels_onehot = load_your_data(...)

# Create attribution object 
big = BA_tensorflow('BIG', use_boundary=True)

# Load the default parameters. You can also modify these parameters if needed. 
# These default parameters should work with one Titan RTX 3080.
parameters = boundary.PARAMETERS
pipeline = boundary.PIPELINE

# Change the backend from 'pytorch' to 'tensorflow'.
parameters.backend = 'tf.keras'

# Compute attribution scores
attr = big.attribute(model,
                     numpy_data_x,
                     labels_onehot,
                     pipline=pipeline,
                     return_dis=False,
                     **parameters)
```

## A Note of DeepLIFT for Pytorch Users
As we are computing DeepLIFT using `Captum`, we notice that if your model is built with `torch.nn.ReLU(in_place=True)` or you use the same `torch.nn.ReLU` object multiple times in your model, `Captum` will fail to attribute. The situation also happens to the `torchvision` models. Therefore, one solution is to disable `inplace` operation and make sure each `ReLU` is only called once in one forward-pass. We provide a `resnet_modified.py` file that contains our fix to this issue. You should replace the `ResNet50` with this one in `torchvision` in order to run DeepLIFT. However, we have not found other attribution methods in `Captum` that have the same issue. 

## Visualization

The quickest way to visualize the attributions is to use [Trulens](https://github.com/truera/trulens). We show an example below for pytorch:

```python
import os
os.environ['TRULENS_BACKEND'] = 'pytorch'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from trulens.visualizations import HeatmapVisualizer

mask_viz = HeatmapVisualizer(blur=7, normalization_type="signed_max")

fig = plt.figure(figsize=(10,10))

_ = mask_viz(attr, img, overlay_opacity=0.5, fig=fig, return_tiled=False)
```

## Citations
```
@misc{wang2021boundary,
      title={Boundary Attributions Provide Normal (Vector) Explanations}, 
      author={Zifan Wang and Matt Fredrikson and Anupam Datta},
      year={2021},
      eprint={2103.11257},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```










