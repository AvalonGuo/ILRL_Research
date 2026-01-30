from typing import Callable,Union
import math
import torch
import torch.nn as nn
import torchvision

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module



class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
    

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers.schedulers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from tqdm import tqdm
import numpy as np
import os
from ilrl.loader import CloseDrawerDataset

# 假设这些函数/类已在其他地方定义：
# - PushTImageDataset
# - get_resnet, replace_bn_with_gn
# - ConditionalUnet1D

class DiffPolicyTrainer:
    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int = 16,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        batch_size: int = 64,
        num_workers: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        num_epochs: int = 100,
        num_diffusion_iters: int = 100,
        vision_backbone: str = 'resnet18',
        lowdim_obs_dim: int = 2,
        action_dim: int = 2,
        device: str = 'cuda',
        use_ema: bool = True,
        ema_power: float = 0.75,
        output_dir: str = "checkpoints"
    ):
        self.dataset_path = dataset_path
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.num_diffusion_iters = num_diffusion_iters
        self.vision_backbone = vision_backbone
        self.lowdim_obs_dim = lowdim_obs_dim
        self.action_dim = action_dim
        self.use_ema = use_ema
        self.ema_power = ema_power
        self.output_dir = output_dir

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化组件
        self.dataset = None
        self.dataloader = None
        self.nets = None
        self.noise_scheduler = None
        self.optimizer = None
        self.lr_scheduler = None
        self.ema = None

        self._setup_dataset()
        self._build_model_and_optimizer()

    def _setup_dataset(self):
        """加载并设置数据集和 dataloader"""
        self.dataset = CloseDrawerDataset(
            dataset_path=self.dataset_path,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )

    def _build_model_and_optimizer(self):
        """构建模型、调度器、优化器和 EMA"""
        from torchvision.models import resnet18
        # --- Vision Encoder ---
        vision_encoder = get_resnet(self.vision_backbone)
        vision_encoder = replace_bn_with_gn(vision_encoder)
        vision_feature_dim = 512  # for ResNet18

        # --- Observation & Action Dimensions ---
        obs_dim = vision_feature_dim + self.lowdim_obs_dim

        # --- Noise Prediction Network ---
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=obs_dim * self.obs_horizon
        )

        self.nets = nn.ModuleDict({
            'vision_encoder': vision_encoder,
            'noise_pred_net': noise_pred_net
        }).to(self.device)

        # --- Diffusion Scheduler ---
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        # --- Optimizer & LR Scheduler ---
        self.optimizer = torch.optim.AdamW(
            self.nets.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.dataloader) * self.num_epochs
        )

        # --- EMA ---
        if self.use_ema:
            self.ema = EMAModel(
                parameters=self.nets.parameters(),
                power=self.ema_power
            )

    def train(self):
        """执行训练循环"""
        print("🚀 Starting training...")
        for epoch in tqdm(range(self.num_epochs), desc="Epoch"):
            epoch_loss = []
            for batch in tqdm(self.dataloader, desc="Batch", leave=False):
                # Device transfer
                nimage = batch['image'][:, :self.obs_horizon].to(self.device)
                nagent_pos = batch['agent_pos'][:, :self.obs_horizon].to(self.device)
                naction = batch['action'].to(self.device)
                B = nagent_pos.shape[0]

                # Encode vision features
                with torch.set_grad_enabled(True):  # Ensure grads for encoder
                    image_features = self.nets['vision_encoder'](
                        nimage.flatten(end_dim=1)
                    )
                image_features = image_features.reshape(*nimage.shape[:2], -1)

                # Concatenate vision + low-dim obs
                obs_features = torch.cat([image_features, nagent_pos], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)  # (B, obs_horizon * obs_dim)

                # Add noise (forward diffusion)
                noise = torch.randn_like(naction, device=self.device)
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (B,), device=self.device
                ).long()
                noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

                # Predict noise
                noise_pred = self.nets['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)

                # Loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # Backprop
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

                # EMA update
                if self.use_ema:
                    self.ema.step(self.nets.parameters())

                # Logging
                loss_val = loss.item()
                epoch_loss.append(loss_val)

            avg_loss = np.mean(epoch_loss)
            print(f"Epoch {epoch + 1}/{self.num_epochs} | Avg Loss: {avg_loss:.6f}")

            # Optional: save checkpoint every few epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1)

        print("✅ Training completed!")

    def save_checkpoint(self, epoch: int):
        """保存模型检查点（含 EMA）"""
        ckpt_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pth")
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.nets.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
        }
        if self.use_ema:
            save_dict['ema_state_dict'] = self.ema.state_dict()
        torch.save(save_dict, ckpt_path)
        print(f"💾 Checkpoint saved to {ckpt_path}")

    def get_ema_model(self):
        """返回使用 EMA 权重的模型（用于推理）"""
        if not self.use_ema:
            return self.nets

        # 创建新模型副本
        ema_nets = nn.ModuleDict({
            'vision_encoder': get_resnet(self.vision_backbone),
            'noise_pred_net': ConditionalUnet1D(
                input_dim=self.action_dim,
                global_cond_dim=(512 + self.lowdim_obs_dim) * self.obs_horizon
            )
        }).to(self.device)
        ema_nets = replace_bn_with_gn(ema_nets['vision_encoder'])  # 注意：这里要重新 apply GN

        # 加载 EMA 参数
        self.ema.copy_to(ema_nets.parameters())  # ← 关键！将 EMA 平均权重复制进去
        return ema_nets
    
if __name__ == "__main__":
    trainer = DiffPolicyTrainer(
        dataset_path='/home/skyfall/RL/ILRL_Research/ilrl/dataset/close_drawer.zarr',
        pred_horizon=8,
        action_dim=3,
        lowdim_obs_dim=6,
        obs_horizon=1,
        action_horizon=4,
        batch_size=2,
        num_epochs=100,
        device='cuda'
    )
    trainer.train()