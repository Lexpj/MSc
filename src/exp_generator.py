FW = "sac"
c = ["SB3", "CleanRL", "TorchRL", FW]
e = ["InvertedPendulum-v4", "HalfCheetah-v4", "Hopper-v4", "Walker2d-v4", "Reacher-v4", "InvertedDoublePendulum-v4", "Ant-v4", "Swimmer-v4"]
HPS = [f"./hps/SB3_{FW}_defaultmujoco.yml", f"./hps/CleanRL_{FW}_defaultmujoco.yml", f"./hps/TorchRL_{FW}_defaultmujoco.yml", f"./hps/{FW}.yml"]

exps = []

# CleanRL failed run on HalfCheetah
exps.append((1, 0, 1))

# Swimmer
for env in [0,2,3,4,5,6]:
    for hps in range(4):
        exps.append((2,hps,env))
    
with open("exp_gen2.sh", "w+") as f:
    f.write("#!/bin/bash\n")
    for i, (fw, hps, env) in enumerate(exps):
        f.write(f"bash job.sh --fw={c[fw]} --alg={FW} --env={e[env]} --steps=3000010 --rep=5 --time=72:00:00 --par=cpu-long --mem=4G --hps='{HPS[hps]}'\n")
