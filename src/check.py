import sys
import argparse
import subprocess
parser = argparse.ArgumentParser(
description="""Slurm Details"""
)
parser.add_argument("--env", required=True, type=str)
parser.add_argument("--rep", required=True, type=int)
parser.add_argument("--steps", required=True, type=int)
parser.add_argument("--fw", required=True, type=str)
parser.add_argument("--alg", required=True, type=str)
parser.add_argument("--hps", required=True, type=str)

args=parser.parse_args()

import os

command = f"conda run -n {args.fw} python ./{args.fw}/train_{args.alg}.py --env={args.env} --alg={args.alg} --steps={1} --rep={1} --fw={args.fw} --hps={args.hps}"
try:
    result = subprocess.check_output(command, shell=True, text=True)
    print("res",result)
    if "ERROR" in result:
        raise Exception(result) # Ironic
    sys.exit(0)
except Exception as e:
    with open("check_log.txt","w+") as f:
        print("err",e)
        f.write(str(e))
    sys.exit(1)
    

