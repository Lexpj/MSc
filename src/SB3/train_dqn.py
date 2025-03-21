import gymnasium as gym
from stable_baselines3 import DQN
import sys

def main(args):
    env = gym.make(args.env)
    if args.log:
        model = DQN("MlpPolicy",env,tensorboard_log=f"./results/{args.alg}_{args.env}_{args.steps}/{args.rep}")
    else:
        model = DQN("MlpPolicy",env)
    model.learn(total_timesteps=args.steps,tb_log_name=f"{args.rep}")
    if args.save_model:
        model.save(f"./results/{args.alg}_{args.env}_{args.steps}/{args.rep}/model")


if __name__ == "__main__":
     import argparse
     
     parser = argparse.ArgumentParser(
     	description="""Details for running this experiment"""
     )
     parser.add_argument("--env", required=True, type=str)
     parser.add_argument("--steps", required=True, type=int)
     parser.add_argument("--rep", required=True, type=int, default=1)
     parser.add_argument("--fw", required=False, type=str, default="SB3")
     parser.add_argument("--alg", required=True, type=str, default="dqn")
     parser.add_argument("--log", required=False, action="store_true")
     parser.add_argument("--save_model", required=False, action="store_true")
     
     args=parser.parse_args()

     
     # Handle cases
     
     main(args)
     
