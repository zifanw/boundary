# Boundary Attribution
Implementation of Boundary Attributions for Normal (Vector) Explanations

![image](media/demo.png)

## Setup
Run `sh install_lib.sh` to install the dependencies. Note that our code is intially supported with Pytorch but we are working on tf.keras version. We will update the libarary as soon as possible. 

The current code is tested using 
```
Python = 3.8.5
Pytorch = 1.6.0
Torchvision = 0.7.0
```

## Compute Boundary Attributions

A complete example is shown in `example.ipynb` and the following code block should work with your own Pytorch model.

```python
import boundary
from boundary.attributions import PT_IntegratedGradient
from boundary import BA_pytorch

# Load your pytorch model
model = load_model(...) 

# Load your data into numpy arrays
numpy_data_x, labels_onehot = load_your_data(...)

# Create attribution object 
big = BA_pytorch('BIG', use_boundary=True)

# Load the default parameters. You can also modify these parameters if needed.
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


## A Note for DeepLIFT
As we are computing DeepLIFT using `Captum`, we notice that if your model is built with `torch.nn.ReLU(in_place=True)` or you use the same `torch.nn.ReLU` object multiple times in your model, `Captum` will fail to attribute. The situation also happens to the `torchvision` models. Therefore, one solution is to disable `inplace` operation and make sure each `ReLU` is only called once in one forward-pass. We provide a `resnet_modified.py` file that contains our fix to this issue. You should replace the `ResNet50` with this one in `torchvision` in order to run DeepLIFT. However, we have not found other attribution methods in `Captum` that have the same issue. 

## Visualization

The quickest way to visualize the attributions is to use `Trulens`. We show an example below:

```python
mask_viz = HeatmapVisualizer(blur=7, normalization_type="signed_max")

fig = plt.figure(figsize=(10,10))

_ = mask_viz(attr, img, overlay_opacity=0.5, fig=fig, return_tiled=False)
```












