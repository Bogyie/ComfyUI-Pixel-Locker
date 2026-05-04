# ComfyUI Pixel Locker

ComfyUI custom nodes for preserving masked pixels while sampling.

## Nodes

- **MaskBuilder**: Converts a base `MASK` into a `PIXEL_LOCK_MASK` with three non-overlapping areas:
  - `hard_keep`: pixels that must be preserved
  - `soft_keep`: boundary pixels that may change slightly
  - `full_edit`: pixels that may change freely
- **PixelLockSampler**: KSampler-style node that locks hard/soft areas through ComfyUI's denoise mask path and applies a final latent lock after sampling.
- **PixelLockComposite**: Final image-space composite. This is the node that guarantees exact RGB preservation in `hard_keep`.

## Recommended Workflow

1. Build a `PIXEL_LOCK_MASK` with **MaskBuilder**.
2. Run **PixelLockSampler** instead of the built-in KSampler.
3. Decode the sampled latent with `VAEDecode`.
4. Run **PixelLockComposite** with the original image, decoded image, and the same `PIXEL_LOCK_MASK`.

`PixelLockSampler` protects structure in latent space. `PixelLockComposite` is required for absolute pixel preservation because VAE encode/decode is not lossless.

For better performance, connect a `VAEEncode` of the original image to PixelLockSampler's optional `original_latent` input. Otherwise connect `original_image` and `vae`; PixelLockSampler will encode the original image only when the lock mask has an active hard or soft area.

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
