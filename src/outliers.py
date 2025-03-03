import numpy as np
import matplotlib.pyplot as plt

res = {
    'Alien': [1850, 1736, 1705],
    'Amidar': [675, 653.53, 585.99],
    'Ant': [2160, 2587, 3800, 5250, 2469, 3258],
    'Assault': [4972, 6791.74, 4878.67],
    'Asterix': [4533, 4820.33, 3738.50],
    'Asteroids': [2098, 1633.67, 1556.90],
    'Atlantis': [2311815.0, 3778458.33, 2036749],
    'BankHeist': [1281, 1195.44, 1213.47],
    'Battlezone': [17367, 24283.75, 19980.0],
    'BeamRider': [1590.0,1915.93,2642.97,1536.20,1591.68,2807,4480,4142,3850,2501.85,2478.44,2835.71],
    'Bowling': [40, 51.62, 59.66],
    'Boxing': [98.31, 95, 92.68, 93.32],
    'Breakout': [335.71,275,414.66,406.57,128.92,138.98,283.0,104,201,132,166,211.24 ,430.09 ,405.73],
    'Centipede': [4386, 3309.34, 3688.54],
    'Chopper': [3516, 5642.83, 816.33],
    'CrazyClimber': [110202.0, 118763.04, 119344.67],
    'DemonAttack': [11378, 29283.83, 13788.43],
    'DoubleDunk': [-15, -6.81, -12.96],
    'Enduro': [758, 1098.9, 1297.23, 986.69],
    'FishingDerby': [18, 21.21, 26.23],
    'Freeway': [33.31, 33, 33.10, 32.97],
    'Frostbite': [314, 1137.34, 933.60],
    'Gopher': [2933, 6505.29, 3672.53],
    'Gravitar': [737, 1099.33, 881.67],
    'HalfCheetah': [2254,2760,3750,1800,6800,2770,5784,1442.64],
    'Hopper': [1622,2508,3350,2330,2500,1703 ,2609,2382.86],
    'Humanoid': [2600, 6750, 787, 716.11],
    'IceHockey': [-4, -4.33, -4.12],
    'InvertedDoublePendulum': [9250, 8000, 9231],
    'InvertedPendulum': [1000, 1000, 1000, 936.09],
    'JamesBond': [561, 496.08, 536.50],
    'Kangaroo': [9929, 6582.12, 5325.33],
    'Krull': [7942, 9718.09, 8737.10],
    'KungFuMaster': [23310, 26000.25, 30451.67],
    'MontezumaRevenge': [42, 0.01, 1.00],
    'MsPacman': [2097, 1699.4, 2345.67, 2152.83],
    'NameThisGame': [6255, 5750.00, 6815.63],
    'Pitfall': [-33, 0.00, -0.76],
    'Pong': [20.52,21,20.36 ,20.512 ,19.78 ,19.79 ,20.3,20.45 ,20.39 ,20.45 ],
    'PrivateEye': [70, 100, 31.83],
    'Qbert': [14293,12341.8,11085,14247,7987,14294,17246.27,15228.25],
    'Reacher': [-7,-7,-4],
    'Riverraid': [8394, 8275.25, 9023.57],
    'RoadRunner': [25076.0, 33040.38, 40125.33],
    'RoboTank': [6,14.43, 16.45],
    'SeaQuest': [1205, 1035.2, 1240.30, 1518.33],
    'SpaceInvaders': [943,1641,671,944,956,1016,1188.82,1019.75],
    'Stargunner': [32689, 43519.12, 44457.67],
    'Swimmer': [128, 108, 67],
    'Tennis': [-15, -17.64, -16.44],
    'TimePilot': [4342.0, 6476.46, 6346.67],
    'Tutankham': [254, 249.05, 190.73],
    'UpNDown': [95445, 487495.41, 156143.70],
    'Venture': [0.0, 0.0, 109.33],
    'VideoPinball': [37389.0, 43133.94, 53121.26],
    'Walker2D':[510,1238,1776,4600,3460,3800,3336 ,3589,2287.95],
    'WizardOfWor': [4185, 6353.58, 5346.33],
    'Zaxxon': [5009,3689.67, 5532.67]
}

notIncluded = [
    'Acrobot', 
    'Bezerk', 
    'Cartpole', 
    'Defender'
    'Finger:turn_hand',
    'Hero',
    'Humanoid:run',
    'MountainCar',
    'Phoenix',
    'Pusher',
    'Skiing',
    'Solaris',
    'Surrond',
    'Walker:run',
    'YarsRevenge'
]

theta = 1.0
Z_SCORES = {}

for key in res.keys():
    data = np.array(res[key])

    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean)/std

    Z_SCORES[key] = z_scores

    # for i in np.where(np.abs(z_scores) > theta)[0]:
    #     print(f"{data[i]} in {key}")

# print("="*20)

for key in res.keys():
    data = np.array(res[key])
    Q1, Q3 = np.percentile(data, [25, 75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - theta * IQR, Q3 + theta * IQR

    for i in np.where((data < lower_bound) | (data > upper_bound))[0]:
        print(f"{data[i]} in {key}")


fig,ax = plt.subplots()
ax.boxplot(Z_SCORES.values(),whis=theta)
ax.set_xticklabels(Z_SCORES.keys(),rotation=90)
plt.show()



