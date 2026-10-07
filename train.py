import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders, load_latents_stats, preprocess_raw_image

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
import math
from torchvision.utils import make_grid

logger = get_logger(__name__)

def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def patchify_latent(x, p):
    """(B,C,H,W) -> (B, (H/p)(W/p), C p p) to token-match encoder features."""
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // p, p, W // p, p)
    return x.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)


@torch.no_grad()
def split_target_by_latent(z, x, mode):
    """
    Per-batch affine split of target features z (B,N,D) against the clean latent x.
      raw:         z unchanged
      residual:    z - proj_X(z)   (LAP-L)
      xpredictive: proj_X(z)       (predictable-only control)
    proj_X(z) = [Xp, 1] @ lstsq([Xp, 1], z), Xp = patchified clean latent matched to z's tokens.
    """
    if mode == "raw":
        return z
    B, N, D = z.shape
    # token grid side = sqrt(N); latent patch size chosen so (H/ps)^2 == N
    side = int(round(N ** 0.5))
    ps = max(1, x.shape[2] // side)

    def _tokens(lat):
        Xp = patchify_latent(lat, ps).float()         # (B, N, C*ps*ps)
        if Xp.shape[1] != N:                          # fallback: interpolate to N tokens
            Xp = torch.nn.functional.interpolate(
                Xp.transpose(1, 2), size=N, mode="linear", align_corners=False).transpose(1, 2)
        return Xp.reshape(B * N, -1)

    zf = z.reshape(B * N, D).float()
    xf = _tokens(x)
    xf = torch.cat([xf, torch.ones(xf.shape[0], 1, device=xf.device)], dim=1)  # bias
    W = torch.linalg.lstsq(xf, zf).solution
    zpred = (xf @ W).reshape(B, N, D).to(z.dtype)
    return zpred if mode == "xpredictive" else (z - zpred)


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device

    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias)
    return z


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == 'none' else args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    if args.enc_type != None:
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
            )
        z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
    else:
        raise NotImplementedError()
    # LAP: frozen local purifier whose prediction is subtracted from the target.
    purifier = None
    if args.target_mode == 'purified':
        from models.purifier import load_purifier
        assert len(encoders) == 1, "purified target mode supports a single encoder"
        purifier, _pck = load_purifier(args.purifier_ckpt, device)
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        sufficiency=args.sufficiency,
        projector_type=('conv' if args.conv_projector else 'mlp'),
        reg_cls=(args.reg_cls_beta > 0),   # REG: extra CLS token in the sequence
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device)
    requires_grad(ema, False)

    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    if args.latents_stats:
        # REPA-E convention: z_norm = (z - bias_e) * scale_e. Ours is z * scale + bias.
        latents_scale, latents_bias = load_latents_stats(args.latents_stats, device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type,
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting,
        proj_weight_schedule=args.proj_weight_schedule,
        proj_tau=args.proj_tau,
        proj_k=args.proj_k,
        struc_coeff=args.struc_coeff,
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Setup data:
    train_dataset = CustomDataset(args.data_dir, latents_dir=args.latents_dir)
    local_batch_size = int(args.batch_size // accelerator.num_processes)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.data_dir})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA",
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    gt_raw_images, gt_xs, _ = next(iter(train_dataloader))
    assert gt_raw_images.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    ys = torch.randint(args.num_classes, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)

    grad_norm = torch.zeros((), device=accelerator.device)  # logged before the first optimizer step under gradient accumulation
    for epoch in range(args.epochs):
        model.train()
        for raw_image, x, y in train_dataloader:
            raw_image = raw_image.to(device)
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            z = None
            labels = y
            with torch.no_grad():
                if args.vae_encode_online:
                    # Cached moments come from SD-VAE; for a swapped tokenizer encode the
                    # raw images here. The posterior is then sampled exactly as before.
                    with torch.no_grad(), accelerator.autocast():
                        _d = vae.encode(raw_image.float() / 127.5 - 1.0).latent_dist
                    x = torch.cat([_d.mean, _d.std], dim=1).float()
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                reg_cls_tok = None   # REG: this batch's DINOv2 CLS token (None = REG off)
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        if args.reg_cls_beta > 0 and encoder_type == 'mae':
                            # REG with an MAE teacher: rows 1.. are the usual MAE patch target
                            # (last block, no final LayerNorm, identical to forward_features);
                            # the CLS token is the final-LayerNorm CLS output, the analogue of
                            # DINOv2's x_norm_clstoken.
                            z = encoder.forward_tokens(raw_image_)
                            reg_cls_tok = encoder.norm(z[:, 0]).float()
                            z = z[:, 1:]
                        else:
                            z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:]
                        if 'dinov2' in encoder_type:
                            # REG: the CLS token comes from the SAME encoder forward as
                            # the patch tokens (their train.py), and becomes both an extra
                            # diffusion target and the 0-th row of the REPA target.
                            if args.reg_cls_beta > 0:
                                reg_cls_tok = z['x_norm_clstoken'].float()
                            z = z['x_norm_patchtokens']
                        if args.target_mode == 'purified':
                            # LAP-N: subtract the frozen local purifier's prediction, in fp32.
                            side = int(round(z.shape[1] ** 0.5))
                            xp = patchify_latent(x, x.shape[2] // side).float()
                            with torch.autocast('cuda', enabled=False):
                                z = z.float() - purifier(xp)
                        elif args.target_mode != 'raw':
                            z = split_target_by_latent(z, x, args.target_mode)
                        if args.spatial_norm:
                            # iREPA-style spatial normalization: per-image, per-dim
                            # standardization across the token axis, applied to the
                            # FINAL target (after any purification/split).
                            zf = z.float()
                            z = ((zf - zf.mean(dim=1, keepdim=True))
                                 / (zf.std(dim=1, keepdim=True) + 1e-6)).to(z.dtype)
                        if reg_cls_tok is not None:
                            # REG aligns an (N+1)-token target: [CLS ; patch tokens].
                            # The LAP target transforms above act on the N patch rows
                            # only; the CLS token has no co-located latent patch.
                            z = torch.cat(
                                [reg_cls_tok.unsqueeze(1).to(z.dtype), z], dim=1)
                        zs.append(z)

            with accelerator.accumulate(model):
                model_kwargs = dict(y=labels)
                loss, proj_loss, suff_loss, align_cos, cls_loss, struc_loss = loss_fn(
                    model, x, model_kwargs, zs=zs, cls_token=reg_cls_tok)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                suff_loss_mean = suff_loss.mean()
                align_cos_mean = align_cos.mean()
                cls_loss_mean = cls_loss.mean()        # REG
                struc_loss_mean = struc_loss.mean()    # sREPA
                loss = (loss_mean + proj_loss_mean * args.proj_coeff + suff_loss_mean * args.lambda_s
                        + cls_loss_mean * args.reg_cls_beta      # REG   (beta, default 0.03)
                        + struc_loss_mean * args.struc_coeff)    # sREPA (lambda_struc, default 2.0)

                ## optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = model.parameters()
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    update_ema(ema, model) # change ema function

            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if accelerator.sync_gradients and global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": accelerator.unwrap_model(model).state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if accelerator.sync_gradients and args.sampling_steps > 0 and (global_step % args.sampling_steps == 0 and global_step > 0):
                import wandb
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        model,
                        xT,
                        ys,
                        num_steps=50,
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                    ).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                 "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")

            if not accelerator.sync_gradients:
                continue  # micro-batch of an accumulated step: log/checkpoint once per optimizer step
            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(),
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "suff_loss": accelerator.gather(suff_loss_mean).mean().detach().item(),
                "align_cos": accelerator.gather(align_cos_mean).mean().detach().item(),
                "cls_loss": accelerator.gather(cls_loss_mean).mean().detach().item(),
                "struc_loss": accelerator.gather(struc_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--config', help='JSON experiment configuration; explicit CLI flags override it')

    # logging:
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--exp-name", type=str, help="defaults to the config file name")
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="none", help="accelerate tracker, e.g. wandb (install separately)")
    parser.add_argument("--sampling-steps", type=int, default=0,
                        help="log EMA sample grids to the tracker every N steps (0 = off)")
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="data/in1k_256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=64)

    # precision
    parser.add_argument("--allow-tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Timestep sampling distribution.")

    # alignment target
    parser.add_argument("--target-mode", type=str, default="raw",
                        choices=["raw", "residual", "xpredictive", "purified"],
                        help="raw=REPA; residual=LAP-L (remove the per-batch affine prediction from "
                             "clean-latent patches); purified=LAP-N (subtract the frozen local "
                             "purifier's prediction); xpredictive=keep only the affine prediction (control).")
    parser.add_argument("--purifier-ckpt", type=str, default=None,
                        help="train_purifier.py checkpoint (required for --target-mode purified).")

    # frozen VAE: cached moments dir, decoder, and per-channel latent statistics. Defaults are SD-VAE.
    parser.add_argument("--latents-dir", type=str, default="vae-sd")
    parser.add_argument("--vae-path", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--latents-stats", type=str, default=None)
    parser.add_argument("--vae-encode-online", action=argparse.BooleanOptionalAction, default=False,
                        help="encode raw images with --vae-path every step instead of using cached moments")

    # REPA-family compositions
    parser.add_argument("--conv-projector", action=argparse.BooleanOptionalAction, default=False,
                        help="iREPA-style convolutional alignment head (3x3 over the "
                             "token grid) instead of the per-token MLP projector.")
    parser.add_argument("--spatial-norm", action=argparse.BooleanOptionalAction, default=False,
                        help="iREPA-style spatial normalization of the alignment target: "
                             "per-image per-dim standardization across tokens, applied "
                             "after any purification/split.")
    # VA-REPA alignment schedule (Stable Velocity defaults: sigmoid, tau=0.7, k=20)
    parser.add_argument("--proj-weight-schedule", type=str, default=None, choices=[None, "sigmoid"])
    parser.add_argument("--proj-tau", type=float, default=0.7)
    parser.add_argument("--proj-k", type=float, default=20.0)
    # REG (Wu et al., NeurIPS 2025, arXiv:2507.01467): prepend a noised CLS token to the
    # latent token sequence and denoise it jointly. REG's default beta is 0.03; 0 = off.
    parser.add_argument("--reg-cls-beta", type=float, default=0.0,
                        help="REG: weight of the extra CLS-token velocity loss (0 = off).")
    # sREPA (Xu et al., arXiv:2605.16949): off-diagonal MSE between the token-token
    # cosine Gram of the teacher target and of the projected student features.
    parser.add_argument("--struc-coeff", type=float, default=0.0,
                        help="sREPA: weight of the structural Gram loss (0 = off; sREPA's default is 2.0).")
    # Prediction-loss intervention: auxiliary velocity prediction from the aligned projection.
    parser.add_argument("--sufficiency", action=argparse.BooleanOptionalAction, default=False,
                        help="Build the auxiliary sufficiency head on the aligned projection.")
    parser.add_argument("--lambda-s", type=float, default=0.0,
                        help="Weight of the auxiliary prediction loss.")

    preliminary, _ = parser.parse_known_args(input_args)
    if preliminary.config:
        with open(preliminary.config) as f:
            defaults = json.load(f)
        known = {a.dest for a in parser._actions}
        unknown = set(defaults) - known
        if unknown:
            parser.error(f'Unknown configuration keys: {sorted(unknown)}')
        parser.set_defaults(**defaults)
    args = parser.parse_args(input_args)

    if not args.exp_name and args.config:
        args.exp_name = Path(args.config).stem
    if not args.exp_name or not args.model:
        parser.error('--exp-name and --model are required (directly or via --config)')
    if args.batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error('batch size and gradient accumulation must be positive')
    if args.vae_path != 'stabilityai/sd-vae-ft-mse' and not args.vae_encode_online:
        parser.error('A swapped VAE requires --vae-encode-online')
    if (args.target_mode == 'purified') != bool(args.purifier_ckpt):
        parser.error('--target-mode purified and --purifier-ckpt must be used together')
    if args.purifier_ckpt:
        # A purifier maps the normalized clean latent to encoder tokens, so it is valid only for
        # the latent space it was fitted on. Checkpoints without a record are SD-VAE-ft-mse.
        from models.purifier import (latent_signature, purifier_latent_signature,
                                     latent_signature_mismatch)
        _psig, _legacy = purifier_latent_signature(
            torch.load(args.purifier_ckpt, map_location="cpu", weights_only=True))
        _rsig = latent_signature(args.vae_path, args.latents_dir, args.vae_encode_online,
                                 args.latents_stats)
        _why = latent_signature_mismatch(_psig, _rsig)
        if _why is not None:
            parser.error(
                "--purifier-ckpt %s was fitted on a different latent space than this run (%s)%s. "
                "Refit it with train_purifier.py --vae-path/--latents-stats/--vae-encode-online "
                "matching this run."
                % (args.purifier_ckpt, _why,
                   " [checkpoint has no latent-space record -> treated as SD-VAE]" if _legacy else ""))
    if args.reg_cls_beta > 0:
        if 'dinov2' not in args.enc_type and args.enc_type not in ('mae-vit-l', 'mae-vit-b'):
            parser.error("--reg-cls-beta needs a DINOv2 or MAE ViT-B/L encoder")
        if len(args.enc_type.split(',')) != 1:
            parser.error("--reg-cls-beta supports a single encoder")
        if args.sufficiency or args.conv_projector:
            parser.error("--reg-cls-beta does not compose with --sufficiency/--conv-projector")
    return args

if __name__ == "__main__":
    args = parse_args()

    main(args)
