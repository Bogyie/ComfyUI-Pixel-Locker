import unittest

import torch
import torch.nn.functional as F

from nodes import MaskBuilder, PixelLockComposite, _denoise_mask_from_lock


class MaskBuilderTests(unittest.TestCase):
    def test_masks_are_non_overlapping_and_exhaustive(self):
        base = torch.zeros((1, 16, 16))
        base[:, 4:12, 4:12] = 1.0

        pixel_lock_mask, hard, soft, edit = MaskBuilder().build(
            base,
            boundary_px=2,
            mask_mode="preserve_mask",
            boundary_mode="centered",
            threshold=0.5,
        )

        hard_b = pixel_lock_mask["hard_keep"]
        soft_b = pixel_lock_mask["soft_keep"]
        edit_b = pixel_lock_mask["full_edit"]

        self.assertTrue(torch.all((hard_b * soft_b) == 0))
        self.assertTrue(torch.all((hard_b * edit_b) == 0))
        self.assertTrue(torch.all((soft_b * edit_b) == 0))
        self.assertTrue(torch.allclose(hard_b + soft_b + edit_b, torch.ones_like(hard_b)))
        self.assertEqual(hard.shape, soft.shape)
        self.assertEqual(soft.shape, edit.shape)


class PixelLockCompositeTests(unittest.TestCase):
    def test_hard_keep_pixels_are_exactly_original(self):
        original = torch.rand((1, 8, 8, 3))
        generated = torch.rand((1, 8, 8, 3))
        base = torch.zeros((1, 8, 8))
        base[:, 2:6, 2:6] = 1.0
        pixel_lock_mask, hard, _soft, _edit = MaskBuilder().build(
            base,
            boundary_px=0,
            mask_mode="preserve_mask",
            boundary_mode="centered",
            threshold=0.5,
        )

        (locked,) = PixelLockComposite().composite(
            original,
            generated,
            pixel_lock_mask,
            boundary_strength=0.35,
        )

        hard_pixels = hard.bool().permute(0, 1, 2).unsqueeze(-1).expand_as(locked)
        self.assertTrue(torch.equal(locked[hard_pixels], original[hard_pixels]))

    def test_soft_keep_uses_bilinear_resize_when_compositing(self):
        original = torch.ones((1, 4, 4, 3))
        generated = torch.zeros((1, 4, 4, 3))
        soft = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        pixel_lock_mask = {
            "hard_keep": torch.zeros_like(soft).unsqueeze(1),
            "soft_keep": soft.unsqueeze(1),
            "full_edit": (1.0 - soft).unsqueeze(1),
        }

        (locked,) = PixelLockComposite().composite(
            original,
            generated,
            pixel_lock_mask,
            boundary_strength=1.0,
        )

        expected_alpha = F.interpolate(soft.unsqueeze(1), size=(4, 4), mode="bilinear", align_corners=False)
        expected = expected_alpha.permute(0, 2, 3, 1).expand_as(locked)
        self.assertTrue(torch.allclose(locked, expected))


class PixelLockSamplerMaskTests(unittest.TestCase):
    def test_lock_alpha_becomes_inverse_denoise_mask(self):
        hard = torch.zeros((1, 1, 4, 4))
        soft = torch.zeros((1, 1, 4, 4))
        hard[:, :, 1:3, 1:3] = 1.0
        soft[:, :, 0, :] = 1.0
        pixel_lock_mask = {
            "hard_keep": hard,
            "soft_keep": soft,
            "full_edit": (1.0 - hard - soft).clamp(0.0, 1.0),
        }

        denoise_mask = _denoise_mask_from_lock(pixel_lock_mask, 4, 4, boundary_strength=0.25)

        self.assertTrue(torch.all(denoise_mask[:, :, 1:3, 1:3] == 0.0))
        self.assertTrue(torch.all(denoise_mask[:, :, 0, :] == 0.75))
        self.assertEqual(float(denoise_mask[:, :, 3, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
