In this section we will introduce a core component of the Diffusers library `UNet2DModel`. This page will closely follow the notebook `hf_diffusers_mnist.ipynb`, so if you're interested in a quick run through and want to play around yourself, then check that out. Here, we will go through the code in more detail and explain the different components of the model, but feel free to refer back to this page.



## `UNet2DModel`
The `UNet2DModel` is a 2D U-Net model that is used for image generation. It abstracts out all of the annoying details of building the model, and allows you to focus on the important parts of your project.


``` python
model = UNet2DModel(
    sample_size=28,                  # (1)
    in_channels=1,                   # (2)
    out_channels=1,                  # (3)
    layers_per_block=1,              # (4)
    block_out_channels=(8, 16, 32),  # (5)
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    num_class_embeds=10,              # (6)
    norm_num_groups=1,                # (7)
)
```

1.  The size of the input image. In this case, it is 28x28 pixels
2.  The number of channels in the input image. In this case, it is 1, as the images are grayscale, but you would use 3 for RGB images
3.  The number of channels in the output image. In this case, it is 1, as the images are grayscale, but you would use 3 for RGB images
4.  The number of layers in each block
5.  The number of output channels in each block - in a convolutional neural network, the number of channels is the number of filters that are applied to the input image.
6.  The number of classes your dataset has if you are doing conditional generation.
7.  The number of groups to use for the normalization layer. This is a hyperparameter that you can tune to improve the performance of your model.

There are a number of additional parameters, but these are the most important ones to understand when you are getting started with the `UNet2DModel`. Let's look at the block types in more detail, and some of the additional paramters

## Further reading
<div class="grid cards" markdown>

-   :fontawesome-solid-book-open:{ .lg .middle } [__CI/CD - Pre-commit resources__](../resources/references.md#pre-commit)

    ---
    Information on GitHub Actions, Black, Flake8, Mypy, Isort, and Git Hooks

</div>