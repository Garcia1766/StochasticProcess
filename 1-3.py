import os, sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import time
import csv

RESULT_DIR = 'results_1-3'
FIG_DIR = 'figs_1-3'

EVAL_NUM = 10000
RANDOM_SEED = 1218

def simulate_once(b: int):
    T = []
    for i in range(EVAL_NUM):
        t = 0
        b_ = b
        while b_ > 0:
            rand_num = random.random()
            if rand_num < 0.25:
                b_ -= 1
            elif rand_num > 0.75:
                b_ += 1
            t += 1
            if t > 5000:  # 截断步数
                break
        T.append(t)
    max_step = np.array(T).max()
    print(f'max_step={max_step}')
    P_T = {0:0}
    for i in range(len(T)):
        if T[i] in P_T:
            P_T[T[i]] += 1
        else:
            P_T[T[i]] = 1
    for key in P_T:
        P_T[key] /= EVAL_NUM
    filename = f'P_T_{b}_{EVAL_NUM}_{RANDOM_SEED}.csv'
    with open(os.path.join(RESULT_DIR, filename),'w',newline='') as f:
        writer = csv.writer(f)
        for row in P_T.items():
            writer.writerow(row)

def draw_for_one_b(b:int):
    t0 = time.time()
    simulate_once(b)
    t1 = time.time()
    print(f'b = {b}, simulate for {EVAL_NUM} times in {t1 - t0} sec')
    filename = f'P_T_{b}_{EVAL_NUM}_{RANDOM_SEED}.csv'
    P_T = pd.read_csv(os.path.join(RESULT_DIR, filename), header=None, index_col=0, squeeze=True).to_dict()
    z_l = [0.5 + i / 200 for i in range(100)]
    fz2 = [(2/z - 1 - 2/z * sqrt(1-z)) ** b for z in z_l]
    fz1 = []
    max_step = np.array(list(P_T.keys())).max() - 1  # 超出截断步数的游走不予考虑
    print(f'max_step={max_step}')
    t2 = time.time()
    for z in z_l:
        fz1_ = 0
        z_ = 1
        for i in range(1, max_step):
            z_ *= z
            if i in P_T:
                fz1_ += P_T[i] * z_
        fz1.append(fz1_)
    t3 = time.time()
    print(f'z-transformation for 100 z in {t3-t2} sec')
    df = pd.DataFrame({'z': z_l, 'f\'(z)': fz1, 'f(z)': fz2})
    filename = f'z_{b}_{EVAL_NUM}_{RANDOM_SEED}.csv'
    df.to_csv(os.path.join(RESULT_DIR, filename), index=False)

    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 150

    x = df.values[:, 0]  # z
    fig, ax1 = plt.subplots()

    y1 = df.values[:, 1]  # fz1
    ax1.plot(x,y1,label='f\'(z)',color='r', linewidth=0.5)
    ax1.set_xlabel("z")
    ax1.set_ylabel('f(z)')

    y2 = df.values[:, 2]  # fz1
    ax1.plot(x,y2,label='f(z)',color='b', linewidth=0.5)

    plt.title(filename[:-4])
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig(os.path.join(FIG_DIR, filename[:-4] + '.png'))


if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    # draw_for_one_b(1)
    for b in range(1, 6):
        draw_for_one_b(b)
    pass
