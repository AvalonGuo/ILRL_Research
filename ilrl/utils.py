import os
import gdown
import h5py
import zarr
from pathlib import Path
import numpy as np
if not hasattr(np, 'product'):
    np.product = np.prod

def download_pushT_data():
    dataset_path = "ilrl/dataset/pusht_cchi_v7_replay.zarr.zip"
    if not os.path.isfile(dataset_path):
        id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
        gdown.download(id=id, output=dataset_path, quiet=False)

def convert_h5_episodes_to_zarr(data_dir: str, output_zarr_path: str):
    """
    将 data_dir 下所有 .h5 文件（每个是一个 episode）合并为 Diffusion Policy 格式的 Zarr。
    
    要求每个 .h5 结构：
      - initial_images/color, depth, flange_pos
      - operations/<timestamp>/color, depth, flange_pos, offset
    """
    h5_files = sorted([f for f in Path(data_dir).glob("*.h5")])
    if not h5_files:
        raise ValueError(f"No .h5 files found in {data_dir}")

    print(f"Found {len(h5_files)} episodes.")

    all_imgs = []
    all_states = []
    all_actions = []
    episode_ends = []

    total_steps = 0

    for h5_file in h5_files:
        print(f"Processing {h5_file.name}...")
        with h5py.File(h5_file, 'r') as f:
            # --- 初始观测 ---
            init_color = f['initial_images']['color'][:]      # (H, W, C)
            init_flange = f['initial_images']['flange_pos'][:]  # (6,)

            # --- 操作序列 ---
            ops = f['operations']
            if len(ops) == 0:
                print(f"Warning: {h5_file} has no operations. Skipping.")
                continue

            timestamps = sorted(ops.keys(), key=float)
            
            # 收集所有颜色和法兰位置（包括初始）
            colors = [init_color]
            flanges = [init_flange]
            offsets = []

            for ts in timestamps:
                grp = ops[ts]
                colors.append(grp['color'][:])
                flanges.append(grp['flange_pos'][:])
                offsets.append(grp['offset'][:])

            # 转为数组
            colors = np.stack(colors)       # (T+1, H, W, C), T = number of operations
            flanges = np.stack(flanges)     # (T+1, 6)
            offsets = np.stack(offsets)     # (T, 3)

            # 验证：offset 数量应等于 colors - 1
            assert len(offsets) == len(colors) - 1, \
                f"Mismatch in {h5_file}: colors={len(colors)}, offsets={len(offsets)}"

            # ✅ 关键修正：只取前 T 个观测（对应 T 个动作）
            valid_colors = colors[:-1]      # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
            valid_flanges = flanges[:-1]    # ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←

            # 添加到全局列表
            all_imgs.append(valid_colors)   # shape (T, H, W, C)
            all_states.append(valid_flanges)  # shape (T, 6)
            all_actions.append(offsets)     # shape (T, 3)

            # 更新 episode_ends（累计总步数 = 当前总动作数）
            total_steps += len(offsets)
            episode_ends.append(total_steps)

    # 拼接所有 episode
    all_imgs = np.concatenate(all_imgs, axis=0)      # (N_total, H, W, C)
    all_states = np.concatenate(all_states, axis=0)  # (N_total, 6)
    all_actions = np.concatenate(all_actions, axis=0)  # (N_total, 3)

    print(f"\nTotal samples: {len(all_imgs)}")
    print(f"Episode ends: {episode_ends}")
    assert len(all_imgs) == len(all_actions), "Length mismatch between img and action!"

    # 创建 Zarr 文件
    root = zarr.open(output_zarr_path, mode='w')
    data_group = root.create_group('data')
    meta_group = root.create_group('meta')

    # 存储图像（保留 uint8，训练时可归一化）
    data_group.create_dataset(
        'img', 
        data=all_imgs, 
        chunks=(min(100, len(all_imgs)), 480, 640, 3),
        dtype='uint8'
    )

    # 存储状态和动作（float32）
    data_group.create_dataset(
        'state',
        data=all_states.astype(np.float32),
        chunks=(min(100, len(all_states)), 6)
    )
    data_group.create_dataset(
        'action',
        data=all_actions.astype(np.float32),
        chunks=(min(100, len(all_actions)), 3)
    )

    # 存储 episode ends
    meta_group.create_dataset(
        'episode_ends',
        data=np.array(episode_ends, dtype=np.int64),
        chunks=False
    )

    print(f"\n✅ Zarr dataset saved to: {output_zarr_path}")
    print("Structure:")
    print("  data/img      ->", all_imgs.shape)
    print("  data/state    ->", all_states.shape)
    print("  data/action   ->", all_actions.shape)
    print("  meta/episode_ends ->", episode_ends)

if __name__ == "__main__":
    # 配置路径
    DATA_DIR = "ilrl/dataset/data_log/close_drawer"
    OUTPUT_ZARR = "ilrl/dataset/close_drawer.zarr"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_ZARR), exist_ok=True)

    # 执行转换
    convert_h5_episodes_to_zarr(DATA_DIR, OUTPUT_ZARR)