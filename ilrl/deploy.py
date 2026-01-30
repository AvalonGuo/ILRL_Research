import torch
import numpy as np
from ilrl.algos.diffusion_policy import DiffPolicyTrainer, get_resnet, replace_bn_with_gn
from torchvision.transforms import Resize
import cv2  # 或你的环境接口

# ===== 配置 =====
CKPT_PATH = "checkpoints/checkpoint_epoch_50.pth"
DEVICE = "cuda"
IMAGE_SIZE = (128, 128)
PRED_HORIZON = 8
OBS_HORIZON = 1
ACTION_HORIZON = 4
ACTION_DIM = 3
LOWDIM_OBS_DIM = 6

resize_transform = Resize(IMAGE_SIZE, antialias=True)

# ===== 加载模型 =====
trainer = DiffPolicyTrainer(
    dataset_path="",  # 不需要真实路径
    pred_horizon=PRED_HORIZON,
    obs_horizon=OBS_HORIZON,
    action_horizon=ACTION_HORIZON,
    action_dim=ACTION_DIM,
    lowdim_obs_dim=LOWDIM_OBS_DIM,
    device=DEVICE
)

# 手动加载 checkpoint
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
trainer.nets.load_state_dict(checkpoint['model_state_dict'])

if trainer.use_ema:
    trainer.ema.load_state_dict(checkpoint['ema_state_dict'])
    policy_nets = trainer.get_ema_model()  # 使用 EMA 模型！
else:
    policy_nets = trainer.nets

policy_nets.eval()  # 切换到 eval 模式！

# ===== 策略函数 =====
def get_action(image: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
    """
    image: (480, 640, 3) uint8 BGR or RGB
    agent_pos: (6,) float32, in [-1, 1]
    returns: (3,) action in [-1, 1]
    """
    with torch.no_grad():
        # 图像预处理
        image = image.astype(np.float32) / 255.0          # [0,1]
        image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
        image = resize_transform(image)                   # (C, 128, 128)
        image = image.unsqueeze(0).unsqueeze(0)           # (1, 1, C, H, W)

        agent_pos = torch.from_numpy(agent_pos).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 6)

        # Encode vision
        image_flat = image.flatten(end_dim=1).to(DEVICE)  # (1, C, H, W)
        vis_feat = policy_nets['vision_encoder'](image_flat)  # (1, 512)
        vis_feat = vis_feat.reshape(1, OBS_HORIZON, -1)

        # Concat with state
        obs_feat = torch.cat([vis_feat, agent_pos.to(DEVICE)], dim=-1)  # (1, 1, 518)
        global_cond = obs_feat.flatten(start_dim=1)  # (1, 518)

        # 生成动作序列（通过 diffusion 采样）
        B = 1
        To = PRED_HORIZON
        Ta = ACTION_HORIZON

        # 初始化随机动作
        noisy_action = torch.randn((B, To, ACTION_DIM), device=DEVICE)

        # 逆向去噪
        for k in trainer.noise_scheduler.timesteps:
            noise_pred = policy_nets['noise_pred_net'](
                noisy_action, 
                torch.full((B,), k, device=DEVICE, dtype=torch.long),
                global_cond=global_cond
            )
            noisy_action = trainer.noise_scheduler.step(
                noise_pred, k, noisy_action
            ).prev_sample

        # 取前 ACTION_HORIZON 步作为输出
        action = noisy_action[0, :Ta].cpu().numpy()  # (Ta, 3)
        return action[0]  # 返回第一步动作

# ===== Rollout 示例 =====
if __name__ == "__main__":
    # 伪环境（替换成你的真实环境）
    for episode in range(10):
        print(f"Episode {episode}")
        # env.reset()
        for step in range(50):
            # obs = env.get_observation()  # dict with 'image' and 'state'
            # image = obs['image']        # (480,640,3) uint8
            # agent_pos = obs['state']    # (6,) in [-1,1]

            # 临时用随机数据测试
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            agent_pos = np.random.uniform(-1, 1, size=6).astype(np.float32)

            action = get_action(image, agent_pos)  # (3,)
            print("Predicted action:", action)

            # env.step(action)
            # if task_success(): break