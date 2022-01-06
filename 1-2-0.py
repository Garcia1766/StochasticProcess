import os, sys
import random
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = 'results_1-2-0'
FIG_DIR = 'figs_1-2-0'

b = 1  # b = (a_2 - a_1)/2 >= 0
p = 0.75  # 1/2 < p <= 1
EVAL_NUM = 1000
RANDOM_SEED = 1217

ET = b / (2*p-1)
print('ET = %.3f' % ET)
varT = p*(1-p)*2*b / (2*p-1)**3
print('varT = %.3f' % varT)
filename = f'{b}_{p}_{EVAL_NUM}_{RANDOM_SEED}.csv'

def simulate():
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
    mu = [T[0]]
    for i in range(1, len(T)):
        mu.append( (mu[i-1]*(i) + T[i]) / (i+1))
    sigma = [0]
    for i in range(1, len(T)):
        sigma.append( ((sigma[i-1] + mu[i-1]**2) * i + T[i]**2) / (i+1) - mu[i]**2)
    itr = [i for i in range(1, EVAL_NUM + 1)]
    df = pd.DataFrame({'itr': itr, 'mu': mu, 'sigma': sigma})
    df.to_csv(os.path.join(RESULT_DIR, filename), index=False)

def draw(filename: str):
    df = pd.read_csv(os.path.join(RESULT_DIR, filename))

    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['savefig.dpi'] = 400
    plt.rcParams['figure.dpi'] = 150

    x = df.values[:, 0]
    fig, ax1 = plt.subplots()

    y1 = df.values[:, 1]
    ax1.plot(x,y1,label='mu',color='r', linewidth=0.5)
    ax1.set_xlabel("itr")
    ax1.set_ylabel('mu')

    y2 = [ET for i in range(EVAL_NUM)]
    ax1.plot(x,y2,label='ET',color='b', linewidth=0.5)

    plt.title(filename[:-4] + '_ET')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig(os.path.join(FIG_DIR, filename[:-4] + '_ET.png'))

    plt.clf()

    x = df.values[:, 0]
    fig, ax1 = plt.subplots()

    y1 = df.values[:, 2]
    ax1.plot(x,y1,label='sigma^2',color='r', linewidth=0.5)
    ax1.set_xlabel("itr")
    ax1.set_ylabel('sigma^2')

    y2 = [varT for i in range(EVAL_NUM)]
    ax1.plot(x,y2,label='varT',color='b', linewidth=0.5)

    plt.title(filename[:-4] + '_varT')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.savefig(os.path.join(FIG_DIR, filename[:-4] + '_varT.png'))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        b = sys.argv[1]  # b = (a_2 - a_1)/2 >= 0
        p = sys.argv[2]  # 1/2 < p <= 1
        EVAL_NUM = sys.argv[3]
        RANDOM_SEED = sys.argv[4]
    random.seed(RANDOM_SEED)
    simulate()
    draw(filename)
    pass
