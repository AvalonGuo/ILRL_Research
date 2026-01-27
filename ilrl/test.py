from ilrl.envs.pusht import PushTImageEnv
import numpy as np


def test_pushTENV():
    env = PushTImageEnv()

    # 1. seed env for initial state.
    # Seed 0-200 are used for the demonstration dataset.
    env.seed(1000)

    # 2. must reset before use
    obs, info = env.reset()

    # 3. 2D positional action space [0,512]
    action = env.action_space.sample()

    # 4. Standard gym step method
    obs, reward, terminated, truncated, info = env.step(action)

    # prints and explains each dimension of the observation and action vectors
    with np.printoptions(precision=4, suppress=True, threshold=5):
        print("obs['image'].shape:", obs['image'].shape, "float32, [0,1]")
        print("obs['agent_pos'].shape:", obs['agent_pos'].shape, "float32, [0,512]")
        print("action.shape: ", action.shape, "float32, [0,512]")

if __name__ == "__main__":
    test_pushTENV()