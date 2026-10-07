"""Numerical and portability checks; no pretrained weights or dataset download."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from dataset import CustomDataset
from loss import SILoss
from models.purifier import LocalPurifier, linear_residual, latent_signature, latent_signature_mismatch
from models.sit import SiT
from samplers import euler_sampler, euler_maruyama_sampler

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_affine_decomposition(self):
        torch.manual_seed(4)
        x = torch.randn(8, 16, 4)
        z = torch.randn(8, 16, 13)
        residual = linear_residual(z, x)
        pred = linear_residual(z, x, 'xpredictive')
        torch.testing.assert_close(residual + pred, z, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(residual.mean((0, 1)), torch.zeros(13), atol=2e-6, rtol=0)
        self.assertLess((x.flatten(0, 1).T @ residual.flatten(0, 1)).abs().max().item(), 1e-4)

    def test_locality_and_latent_guard(self):
        p = LocalPurifier(in_ch=4, hidden=8, out_dim=7, rf=3)
        x = torch.randn(2, 64, 4)
        a = p(x)
        x[:, -1] += 20
        b = p(x)
        torch.testing.assert_close(a[:, 0], b[:, 0], atol=0, rtol=0)
        self.assertIsNone(latent_signature_mismatch(latent_signature(), latent_signature()))
        self.assertIsNotNone(latent_signature_mismatch(latent_signature(), latent_signature('other/vae', 'vae-sd', True)))

    def test_manifest_pairs_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'images').mkdir(); (p/'vae-sd').mkdir()
            (p/'vae-sd/dataset.json').write_text(json.dumps({'labels': [['one.npy', 3]]}))
            np.save(p/'vae-sd/one.npy', np.zeros((8, 4, 4), np.float32))
            Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(p/'images/one.png')
            ds = CustomDataset(d)
            x, m, y = ds[0]
            self.assertEqual(tuple(x.shape), (3, 32, 32)); self.assertEqual(int(y), 3)
            (p/'images/one.png').write_bytes(b'not an image')
            with self.assertRaises(Exception):
                ds[0]  # Never silently replace a held-out or training image.

    def test_recipe_backward_and_reg_sampling(self):
        torch.set_num_threads(2)
        for recipe in ['repa', 'irepa', 'varepa', 'reg', 'srepa', 'prediction']:
            torch.manual_seed(0)
            m = SiT(input_size=8, patch_size=2, hidden_size=64, depth=3, num_heads=4,
                    encoder_depth=2, decoder_hidden_size=64, z_dims=[16], projector_dim=32, num_classes=1000,
                    projector_type='conv' if recipe == 'irepa' else 'mlp',
                    reg_cls=recipe == 'reg', sufficiency=recipe == 'prediction',
                    fused_attn=False, qk_norm=False)
            x = torch.randn(2, 4, 8, 8)
            cls = torch.randn(2, 16) if recipe == 'reg' else None
            z = torch.randn(2, 17 if cls is not None else 16, 16)
            f = SILoss(struc_coeff=2 if recipe == 'srepa' else 0,
                       proj_weight_schedule='sigmoid' if recipe == 'varepa' else None)
            losses = f(m, x, {'y': torch.tensor([1, 2])}, [z], cls_token=cls)
            total = sum(v.mean() for i, v in enumerate(losses) if i != 3)
            self.assertTrue(torch.isfinite(total), recipe)
            total.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None), recipe)
            if recipe == 'reg':
                m.eval()
                with torch.no_grad():
                    for sampler in [euler_sampler, euler_maruyama_sampler]:
                        out = sampler(m, x, torch.tensor([1, 2]), num_steps=3, cfg_scale=1.5)
                        self.assertEqual(out.shape, x.shape)
                        self.assertTrue(torch.isfinite(out).all())

    def test_config_manifest(self):
        import train
        registry = json.loads((ROOT/'configs/experiments.json').read_text())
        for name, entry in registry.items():
            path = ROOT/entry['config']
            self.assertTrue(path.is_file(), name)
            c = json.loads(path.read_text())
            self.assertFalse(any(isinstance(v, str) and v.startswith(('/mnt/', '/home/')) for v in c.values()))
            argv = ['--config', str(path), '--exp-name', name, '--seed', str(entry['training_seed']),
                    '--max-train-steps', str(entry['training_steps'])]  # as reproduce.py train
            if c.get('purifier_ckpt') and not (ROOT/c['purifier_ckpt']).is_file():
                argv += ['--target-mode', 'raw', '--purifier-ckpt', '']  # small assets are not in Git
            args = train.parse_args(argv)
            self.assertEqual(args.batch_size, 64, name)


if __name__ == '__main__':
    unittest.main()
