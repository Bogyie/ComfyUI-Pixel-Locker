# ComfyUI Pixel Locker

ComfyUI custom nodes for preserving masked pixels while sampling.

## Nodes

- **MaskBuilder**: Converts a base `MASK` into a `PIXEL_LOCK_MASK` with three non-overlapping areas:
  - `hard_keep`: pixels that must be preserved
  - `soft_keep`: boundary pixels that may change slightly
  - `full_edit`: pixels that may change freely
- **PixelLockSampler**: KSampler-style inpaint sampler that starts from the original latent and adds noise only to editable mask regions.
- **PixelLockComposite**: Final image-space composite. This is the node that guarantees exact RGB preservation in `hard_keep`.
- **PixelLockDecodeComposite**: Convenience node that decodes a sampled latent and immediately applies the final pixel composite.

## Recommended Workflow

1. Build a `PIXEL_LOCK_MASK` with **MaskBuilder**.
2. Run **PixelLockSampler** instead of the built-in KSampler.
3. Decode and restore pixels with **PixelLockDecodeComposite**.

`PixelLockSampler` protects structure in latent space. `PixelLockComposite` is required for absolute pixel preservation because VAE encode/decode is not lossless.

For better performance, connect a `VAEEncode` of the original image to PixelLockSampler's optional `original_latent` input. Otherwise connect `original_image` and `vae`; PixelLockSampler will encode the original image only when the lock mask has an active hard or soft area.

When replacing clothing or other large regions, start with `edit_strength` around `0.75-0.9`. PixelLockSampler does not use an incoming empty/base latent; it always starts from `original_latent` or an encoded `original_image`, then denoises only `full_edit` and `soft_keep` according to the mask strengths.

## Mask Semantics

`MaskBuilder` supports two base mask modes:

- `preserve_mask`: white means preserve.
- `edit_mask`: white means edit.

Boundary modes:

- `centered`: soft band is created both inside and outside the preservation edge.
- `inward`: soft band is only inside the preserved region.
- `outward`: soft band is only outside the preserved region.

## Installation

Place this folder in `ComfyUI/custom_nodes/ComfyUI-Pixel-Locker`, then restart ComfyUI.
