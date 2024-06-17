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

There are a number of additional parameters, but these are the most important ones to understand when you are getting started with the `UNet2DModel`. There are also many different types of blocks that you can use in the `UNet2DModel`, and you can customize the model to suit your needs. For more information on arguments and blocks check out the links below.

Previously, we built all of our components from scratch, but now we can use the `UNet2DModel` to abstract out all of the annoying details of building the model. This allows us to focus on the important parts of our project, such as the parameters for the diffusion process.

## The Schedulers
To define the schedulers, we use the `DDPMScheduler` and `get_cosine_schedule_with_warmup` functions. The `DDPMScheduler` is a scheduler that is used to schedule the noise level in the diffusion process, and the `get_cosine_schedule_with_warmup` function is used to schedule the learning rate of the optimizer. Essentially, it is possible for the initial training to skew highly towards any strong features that might appear at the start of training, or even worse, towards noise. Warm-up is a technique that helps to mitigate this issue by gradually increasing the learning rate from zero to the desired value over a few steps. This is usually around an epoch or so, but we've reduced it to 50 steps for this example.

```python
num_epochs = 3

noise_scheduler = DDPMScheduler(num_train_timesteps=200,    # (1)
                                beta_start = 0.0001,        # (2)
                                beta_end = 0.02,            # (3)
                                beta_schedule = 'linear',   # (4)
                                prediction_type = 'epsilon' # (5)
                                )
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

num_train_steps = len(train_loader) * epochs

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=50,
    num_training_steps=(num_train_steps),
)
```

1.  How many timesteps do we want to include in the diffusion process.
2.  The starting value of the noise level.
3.  The ending value of the noise level.
4.  The schedule for the noise level. We are using a linear schedule, but there are none linear options.
5.  The type of prediction that we are using. `'epsilon'` means we are predicting the noise level of the diffusion process, but we could also predict the noisy sample using `'sample'` directly.

### The Diffusion Scheduler
We are using the `DDPMScheduler` and checking out the [documentation for this scheduler](https://huggingface.co/docs/diffusers/en/api/schedulers/ddpm) reveals that we can define things like the beta schedule and the type of predictions we can make. This is not the only scheduler available such as DDIM and Flow-based solvers. For a full list, see the [documentation](https://huggingface.co/docs/diffusers/en/api/schedulers/overview).


## Further reading
<div class="grid cards" markdown>

-   :fontawesome-solid-book-open:{ .lg .middle } [__CI/CD - Pre-commit resources__](../resources/references.md#pre-commit)

    ---
    Information on GitHub Actions, Black, Flake8, Mypy, Isort, and Git Hooks

</div>