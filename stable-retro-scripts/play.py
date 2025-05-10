import argparse
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.monitor import Monitor

def make_env(game, state, rank):
    def _init():
        env = retro.make(
            game=game,
            state=state,
            obs_type=retro.Observations.IMAGE
        )
        env = Monitor(env)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model .zip file')
    parser.add_argument('--env', type=str, default='MortalKombatII-Genesis',
                       help='Game environment ID')
    parser.add_argument('--state', type=str, required=True,
                       help='Initial state name')
    parser.add_argument('--num_env', type=int, default=1,
                       help='Number of parallel environments')
    args = parser.parse_args()

    # Create environment
    venv = SubprocVecEnv([
        make_env(args.env, args.state, i)
        for i in range(args.num_env)
    ])
    venv = VecTransposeImage(venv)
    venv = VecFrameStack(venv, n_stack=4)

    # Load trained model
    model = PPO.load(args.model_path, env=venv)

    # Run the model
    obs = venv.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = venv.step(action)
        venv.render()
        if dones.any():
            obs = venv.reset()

if __name__ == "__main__":
    main()
