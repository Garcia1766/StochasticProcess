import os, sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = 'results_1-2'
FIG_DIR = 'figs_1-2'

b = 1  # b = (a_2 - a_1)/2 >= 1
p = 0.75  # 1/2 < p < 1
EVAL_NUM = 10000
RANDOM_SEED = 1217

def simulate_once(b: int, p: float):
    T = []
    for i in range(EVAL_NUM):
        t = 0
        b_ = b
        while b_ > 0:
            rand_num = random.random()
            if rand_num < p*p:
                b_ -= 1
            elif rand_num > p*(2-p):
                b_ += 1
            t += 1
        T.append(t)
    T = np.array(T)
    mu = T.mean()
    sigma = T.var()
    return mu, sigma

def simulate_for_p(b:int):
    p_l, mu_l, sigma_l = [], [], []
    for i in range(1, 101):
        p = 0.5 + 0.5 * i / 100
        mu, sigma = simulate_once(b, p)
        p_l.append(p)
        mu_l.append(mu)
        sigma_l.append(sigma)
    df = pd.DataFrame({'p': p_l, 'mu': mu_l, 'sigma': sigma_l})
    filename = f'{b}_{EVAL_NUM}_{RANDOM_SEED}.csv'
    df.to_csv(os.path.join(RESULT_DIR, filename), index=False)

    df = pd.read_csv(os.path.join(RESULT_DIR, filename))

    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 150

    x = df.values[:, 0]  # p
    fig, ax1 = plt.subplots()

    y1 = df.values[:, 1]  # mu
    ax1.plot(x,y1,label='mu',color='r', linewidth=0.5)
    ax1.set_xlabel("p")
    ax1.set_ylabel('mu')

    y2 = [b / (2*p-1) for p in x]
    ax1.plot(x,y2,label='ET',color='b', linewidth=0.5)

    plt.title(filename[:-4] + '_ET')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig(os.path.join(FIG_DIR, filename[:-4] + '_ET.png'))

    plt.clf()

    fig, ax1 = plt.subplots()

    y1 = df.values[:, 2]  # sigma^2
    ax1.plot(x,y1,label='sigma^2',color='r', linewidth=0.5)
    ax1.set_xlabel("p")
    ax1.set_ylabel('sigma^2')

    y2 = [p*(1-p)*2*b / (2*p-1)**3 for p in x]
    ax1.plot(x,y2,label='varT',color='b', linewidth=0.5)

    plt.title(filename[:-4] + '_varT')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig(os.path.join(FIG_DIR, filename[:-4] + '_varT.png'))

def simulate_for_b(p:float):
    b_l, mu_l, sigma_l = [], [], []
    for i in range(1, 101):
        b = i
        mu, sigma = simulate_once(b, p)
        b_l.append(b)
        mu_l.append(mu)
        sigma_l.append(sigma)
    df = pd.DataFrame({'b': b_l, 'mu': mu_l, 'sigma': sigma_l})
    filename = f'{p}_{EVAL_NUM}_{RANDOM_SEED}.csv'
    df.to_csv(os.path.join(RESULT_DIR, filename), index=False)

    df = pd.read_csv(os.path.join(RESULT_DIR, filename))

    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 150

    x = df.values[:, 0]  # b
    fig, ax1 = plt.subplots()

    y1 = df.values[:, 1]  # mu
    ax1.plot(x,y1,label='mu',color='r', linewidth=0.5)
    ax1.set_xlabel("b")
    ax1.set_ylabel('mu')

    y2 = [b / (2*p-1) for b in x]
    ax1.plot(x,y2,label='ET',color='b', linewidth=0.5)

    plt.title(filename[:-4] + '_ET')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig(os.path.join(FIG_DIR, filename[:-4] + '_ET.png'))

    plt.clf()

    fig, ax1 = plt.subplots()

    y1 = df.values[:, 2]  # sigma^2
    ax1.plot(x,y1,label='sigma^2',color='r', linewidth=0.5)
    ax1.set_xlabel("p")
    ax1.set_ylabel('sigma^2')

    y2 = [p*(1-p)*2*b / (2*p-1)**3 for b in x]
    ax1.plot(x,y2,label='varT',color='b', linewidth=0.5)

    plt.title(filename[:-4] + '_varT')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig(os.path.join(FIG_DIR, filename[:-4] + '_varT.png'))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        b = sys.argv[1]  # b = (a_2 - a_1)/2 >= 1
        p = sys.argv[2]  # 1/2 < p < 1
        EVAL_NUM = sys.argv[3]
        RANDOM_SEED = sys.argv[4]
    random.seed(RANDOM_SEED)
    simulate_for_p(b)
    simulate_for_b(p)
    pass
