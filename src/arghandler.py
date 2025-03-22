import yaml
import os
import argparse

def handle(args):
    parser = argparse.ArgumentParser(
       description="""Details for running this experiment"""
    )
    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--rep", required=False, type=int, default=1)
    parser.add_argument("--fw", required=False, type=str, default="SB3")
    parser.add_argument("--alg", required=False, type=str, default="ppo")
    parser.add_argument("--hps", required=False, type=str)
    parser.add_argument("--log", required=False, action="store_true")
    parser.add_argument("--save_model", required=False, action="store_true")
    
    args=parser.parse_args()
    if not args.hps: #Note that this is handled automatically when called via a SLURM job
        args.hps = f"./hps/{args.fw}_{args.alg}_default.yml"
    print(f"Using hyperparameter file {args.hps}")
    try:
        with open(args.hps) as f:
           config = yaml.safe_load(f)
    except:
        with open("../"+args.hps) as f:
            config = yaml.safe_load(f)
    config['hps'] = args.hps.split("/")[-1].split(".")[0].split("_")[-1]
    
    
    
    # Last folder (for repeated experiments)
    try: 
        config['lastfolder'] = 1+max([int(x.split("_")[1]) for x in os.listdir(f"./{args.fw}/results/{args.alg}_{args.env}_{args.steps}/{config['hps']}")])
    except Exception as e:
        if args.steps != 1:
            print(e)
            print("Using initial folder")
        config['lastfolder'] = 1
    
    config['env'] = args.env
    config['steps'] = args.steps
    config['rep'] = args.rep
    config['fw'] = args.fw
    config['alg'] = args.alg
    config['log'] = args.log
    config['save_model'] = args.save_model
    
    
    return config
