from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


try:
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview

    SAMPLER_NAMES = comfy.samplers.KSampler.SAMPLERS
    SCHEDULER_NAMES = comfy.samplers.KSampler.SCHEDULERS
except Exception:
    comfy = None
    latent_preview = None
    SAMPLER_NAMES = ["euler"]
    SCHEDULER_NAMES = ["normal"]


PIXEL_LOCK_MASK = "PIXEL_LOCK_MASK"
CATEGORY = "Pixel Locker"


def _mask_to_bchw(mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError("Expected mask to be a torch.Tensor.")

    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4:
        if mask.shape[1] != 1:
            mask = mask[:, :1]
    else:
        raise ValueError(f"Unsupported MASK shape: {tuple(mask.shape)}")

    return mask.float().clamp(0.0, 1.0)


def _bchw_to_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.squeeze(1).float().clamp(0.0, 1.0)


def _image_to_bhwc(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("Expected image to be a torch.Tensor.")
    if image.ndim != 4:
        raise ValueError(f"Expected IMAGE shape [B,H,W,C], got {tuple(image.shape)}.")
    return image.float().clamp(0.0, 1.0)


def _resize_mask(mask: torch.Tensor, height: int, width: int, mode: str = "bilinear") -> torch.Tensor:
    mask = _mask_to_bchw(mask)
    if mask.shape[-2:] == (height, width):
        return mask

    if mode == "nearest":
        return F.interpolate(mask, size=(height, width), mode="nearest").clamp(0.0, 1.0)
    return F.interpolate(mask, size=(height, width), mode="bilinear", align_corners=False).clamp(0.0, 1.0)


def _expand_batch(tensor: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    raise ValueError(f"{name} batch size {tensor.shape[0]} does not match target batch size {batch}.")


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    return 1.0 - _dilate(1.0 - mask, radius)


def _make_pixel_lock_mask(hard_keep: torch.Tensor, soft_keep: torch.Tensor, full_edit: torch.Tensor) -> dict[str, torch.Tensor]:
    hard_keep = _mask_to_bchw(hard_keep)
    soft_keep = _mask_to_bchw(soft_keep)
    full_edit = _mask_to_bchw(full_edit)
    return {
        "hard_keep": hard_keep,
        "soft_keep": soft_keep,
        "full_edit": full_edit,
    }


def _extract_pixel_lock_mask(pixel_lock_mask: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(pixel_lock_mask, dict):
        raise TypeError("PIXEL_LOCK_MASK must be produced by Pixel Locker MaskBuilder.")

    missing = {"hard_keep", "soft_keep", "full_edit"} - set(pixel_lock_mask.keys())
    if missing:
        raise ValueError(f"PIXEL_LOCK_MASK is missing fields: {', '.join(sorted(missing))}.")

    return (
        _mask_to_bchw(pixel_lock_mask["hard_keep"]),
        _mask_to_bchw(pixel_lock_mask["soft_keep"]),
        _mask_to_bchw(pixel_lock_mask["full_edit"]),
    )


def _latent_lock_alpha(pixel_lock_mask: dict[str, Any], height: int, width: int, boundary_strength: float) -> torch.Tensor:
    hard_keep, soft_keep, _full_edit = _extract_pixel_lock_mask(pixel_lock_mask)
    hard_keep = _resize_mask(hard_keep, height, width, mode="nearest")
    soft_keep = _resize_mask(soft_keep, height, width, mode="bilinear")
    boundary_strength = float(max(0.0, min(1.0, boundary_strength)))
    return (hard_keep + soft_keep * boundary_strength).clamp(0.0, 1.0)


def _denoise_mask_from_lock(pixel_lock_mask: dict[str, Any], height: int, width: int, boundary_strength: float) -> torch.Tensor:
    return (1.0 - _latent_lock_alpha(pixel_lock_mask, height, width, boundary_strength)).clamp(0.0, 1.0)


def _has_mask_effect(mask: torch.Tensor, epsilon: float = 1e-6) -> bool:
    return bool(mask.detach().abs().max().item() > epsilon)


def _is_full_denoise_mask(mask: torch.Tensor, epsilon: float = 1e-6) -> bool:
    return bool((1.0 - mask.detach()).abs().max().item() <= epsilon)


def _composite_alpha(pixel_lock_mask: dict[str, Any], height: int, width: int, boundary_strength: float) -> torch.Tensor:
    hard_keep, soft_keep, _full_edit = _extract_pixel_lock_mask(pixel_lock_mask)
    hard_keep = _resize_mask(hard_keep, height, width, mode="nearest")
    soft_keep = _resize_mask(soft_keep, height, width, mode="bilinear")
    boundary_strength = float(max(0.0, min(1.0, boundary_strength)))
    return (hard_keep + soft_keep * boundary_strength).clamp(0.0, 1.0)


def _apply_latent_lock(current: torch.Tensor, original: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    original = original.to(device=current.device, dtype=current.dtype)
    alpha = alpha.to(device=current.device, dtype=current.dtype)
    original = _expand_batch(original, current.shape[0], "original latent")
    alpha = _expand_batch(alpha, current.shape[0], "pixel lock mask")

    if original.shape[-2:] != current.shape[-2:]:
        original = F.interpolate(original, size=current.shape[-2:], mode="bilinear", align_corners=False)

    return current * (1.0 - alpha) + original * alpha


def _encode_original_latent(vae: Any, original_image: torch.Tensor, latent_image: dict[str, Any]) -> torch.Tensor:
    if "samples" not in latent_image:
        raise ValueError("LATENT input is missing the 'samples' tensor.")

    encoded = vae.encode(original_image)
    target = latent_image["samples"]
    if encoded.shape[-2:] != target.shape[-2:]:
        encoded = F.interpolate(encoded, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return encoded


def _sample_with_lock(
    model: Any,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    positive: Any,
    negative: Any,
    latent: dict[str, Any],
    original_samples: torch.Tensor | None,
    original_image: torch.Tensor | None,
    vae: Any | None,
    pixel_lock_mask: dict[str, Any],
    boundary_strength: float,
    denoise: float,
) -> tuple[dict[str, Any]]:
    if comfy is None or latent_preview is None:
        raise RuntimeError("PixelLockSampler must be run inside ComfyUI.")

    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
    )

    batch_inds = latent["batch_index"] if "batch_index" in latent else None
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
    base_callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    alpha = _latent_lock_alpha(pixel_lock_mask, latent_image.shape[-2], latent_image.shape[-1], boundary_strength)
    lock_has_effect = _has_mask_effect(alpha)
    if lock_has_effect and original_samples is None:
        if original_image is None or vae is None:
            raise ValueError("PixelLockSampler needs original_image and vae when original_latent is not connected.")
        original_samples = _encode_original_latent(vae, _image_to_bhwc(original_image), latent)

    locked_latent_image = _apply_latent_lock(latent_image, original_samples, alpha) if lock_has_effect else latent_image
    lock_denoise_mask = (1.0 - alpha).clamp(0.0, 1.0)
    lock_denoise_mask = _expand_batch(lock_denoise_mask, latent_image.shape[0], "pixel lock denoise mask")
    noise_mask = None if _is_full_denoise_mask(lock_denoise_mask) else lock_denoise_mask
    if "noise_mask" in latent:
        existing_noise_mask = _resize_mask(latent["noise_mask"], latent_image.shape[-2], latent_image.shape[-1])
        existing_noise_mask = _expand_batch(existing_noise_mask, latent_image.shape[0], "latent noise mask")
        noise_mask = existing_noise_mask * lock_denoise_mask

    def callback(step: int, denoised: torch.Tensor, current: torch.Tensor, total_steps: int) -> None:
        base_callback(step, denoised, current, total_steps)

    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        locked_latent_image,
        denoise=denoise,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        noise_mask=None if noise_mask is None else noise_mask.clamp(0.0, 1.0),
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    if lock_has_effect:
        samples = _apply_latent_lock(samples, original_samples, alpha)

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out["samples"] = samples
    return (out,)


class MaskBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_mask": ("MASK", {"tooltip": "Mask that describes the preserved or edited region."}),
                "boundary_px": ("INT", {"default": 8, "min": 0, "max": 512, "step": 1}),
                "mask_mode": (["preserve_mask", "edit_mask"], {"default": "preserve_mask"}),
                "boundary_mode": (["centered", "inward", "outward"], {"default": "centered"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = (PIXEL_LOCK_MASK, "MASK", "MASK", "MASK")
    RETURN_NAMES = ("pixel_lock_mask", "hard_keep", "soft_keep", "full_edit")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self,
        base_mask: torch.Tensor,
        boundary_px: int,
        mask_mode: str,
        boundary_mode: str,
        threshold: float,
    ):
        mask = (_mask_to_bchw(base_mask) >= threshold).float()
        preserve = mask if mask_mode == "preserve_mask" else 1.0 - mask
        preserve = preserve.clamp(0.0, 1.0)

        radius = int(max(0, boundary_px))
        if radius == 0:
            hard_keep = preserve
            soft_keep = torch.zeros_like(preserve)
            full_edit = 1.0 - hard_keep
        elif boundary_mode == "inward":
            hard_keep = _erode(preserve, radius)
            soft_keep = (preserve - hard_keep).clamp(0.0, 1.0)
            full_edit = 1.0 - preserve
        elif boundary_mode == "outward":
            hard_keep = preserve
            dilated = _dilate(preserve, radius)
            soft_keep = (dilated - hard_keep).clamp(0.0, 1.0)
            full_edit = 1.0 - dilated
        else:
            eroded = _erode(preserve, radius)
            dilated = _dilate(preserve, radius)
            hard_keep = eroded
            soft_keep = (dilated - eroded).clamp(0.0, 1.0)
            full_edit = 1.0 - dilated

        hard_keep = hard_keep.clamp(0.0, 1.0)
        soft_keep = (soft_keep * (1.0 - hard_keep)).clamp(0.0, 1.0)
        full_edit = (1.0 - hard_keep - soft_keep).clamp(0.0, 1.0)
        soft_keep = (1.0 - hard_keep - full_edit).clamp(0.0, 1.0)

        pixel_lock_mask = _make_pixel_lock_mask(hard_keep, soft_keep, full_edit)
        return (
            pixel_lock_mask,
            _bchw_to_mask(hard_keep),
            _bchw_to_mask(soft_keep),
            _bchw_to_mask(full_edit),
        )


class PixelLockSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (SAMPLER_NAMES,),
                "scheduler": (SCHEDULER_NAMES,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "pixel_lock_mask": (PIXEL_LOCK_MASK,),
                "boundary_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "original_image": ("IMAGE",),
                "vae": ("VAE",),
                "original_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(
        self,
        model: Any,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        positive: Any,
        negative: Any,
        latent_image: dict[str, Any],
        pixel_lock_mask: dict[str, Any],
        boundary_strength: float,
        denoise: float,
        original_image: torch.Tensor | None = None,
        vae: Any | None = None,
        original_latent: dict[str, Any] | None = None,
    ):
        original_samples = original_latent["samples"] if original_latent is not None else None

        return _sample_with_lock(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            original_samples,
            original_image,
            vae,
            pixel_lock_mask,
            boundary_strength,
            denoise,
        )


class PixelLockComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "generated_image": ("IMAGE",),
                "pixel_lock_mask": (PIXEL_LOCK_MASK,),
                "boundary_strength": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = CATEGORY

    def composite(
        self,
        original_image: torch.Tensor,
        generated_image: torch.Tensor,
        pixel_lock_mask: dict[str, Any],
        boundary_strength: float,
    ):
        original = _image_to_bhwc(original_image)
        generated = _image_to_bhwc(generated_image)

        if original.shape[1:3] != generated.shape[1:3]:
            raise ValueError(
                "PixelLockComposite requires original_image and generated_image to have the same H/W. "
                "Resize or crop before compositing; otherwise exact pixel preservation is not well-defined."
            )

        hard_keep, _soft_keep, _full_edit = _extract_pixel_lock_mask(pixel_lock_mask)
        alpha = _composite_alpha(pixel_lock_mask, generated.shape[1], generated.shape[2], boundary_strength)
        alpha = _expand_batch(alpha, generated.shape[0], "pixel lock mask").permute(0, 2, 3, 1)
        original = _expand_batch(original, generated.shape[0], "original image")

        image = generated * (1.0 - alpha) + original * alpha
        hard = _resize_mask(hard_keep, generated.shape[1], generated.shape[2], mode="nearest")
        hard = _expand_batch(hard, generated.shape[0], "hard keep mask").permute(0, 2, 3, 1).bool()
        image = torch.where(hard, original, image)
        return (image.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {
    "PixelLockerMaskBuilder": MaskBuilder,
    "PixelLockerPixelLockSampler": PixelLockSampler,
    "PixelLockerPixelLockComposite": PixelLockComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelLockerMaskBuilder": "MaskBuilder",
    "PixelLockerPixelLockSampler": "PixelLockSampler",
    "PixelLockerPixelLockComposite": "PixelLockComposite",
}
