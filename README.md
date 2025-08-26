# DINOv3

DINOv3 seems to be a great model.

You can download the models if you share your contact info with Meta.
Once you do, you can load them using `torch.hub.load`.
For instance, this loads the ViT-B/16.

```py
vit = torch.hub.load(
    "facebookresearch/dinov3", "dinov3_vitb16", source="github", weights="/PATH/TO/MODELS/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)
```

However, this only will give you the [CLS] token representation.
If you want to do more stuff, you will have to hack at the model code.

To make this easier, I have two single-file implemenations of DINOv3, in both PyTorch and Jax/Equinox.
I have compared outputs against the reference implementation and they are identical for the [CLS] token up to floating-point errors for ViT-S/16, ViT-B/16 and ViT-L/16.

# PyTorch

After downloading the checkpoints to a directory $CKPTS, you can use `dinov3_pt.load()`

```py

```

# Jax

After downloading the checkpoints to a directory $CKPTS, you need to convert them to the Equinox model using `dinov3_jax.py`.

```sh
uv run --with tyro dinov3_jax.py \
  --ckpt $CKPTS/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dump-to $CKPTS-jax
```

Then you can use `dinov3_jax.load()` to load them.

```py
vit = dinov3_jax.load(f"$CKPTS-jax/dinov3_vitl16.eqx")
```

Then you can use this ViT as an equinox module.

If you want to change the model, you can update the forward pass to return something different.
If you want to update parameter shapes, you might need to convert the PyTorch checkpoints to Equinox checkpoints using the `_convert()` function again.

# Testing

```sh
uv run pytest . \
  --pt-ckpts $CKPTS \
  --jax-ckpts $CKPTS-jax
```
