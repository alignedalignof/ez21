import random
from collections import defaultdict
import operator
import copy

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm

HIT = "hit"
STICK = "stick"
N0 = 100
FEATURES = 36

DECK_MAX = 21
CARD_MAX = 10
CARD_MIN = 1
DEALER_STICK = 17

def draw_card():
    n = random.choice([i for i in range(CARD_MIN, CARD_MAX + 1)])
    return n if random.choice(["red", "black", "black"]) == "black" else -n

def draw_blacks():
    player = abs(draw_card())
    dealer = abs(draw_card())
    return player, dealer

def is_bust(deck):
    return deck > DECK_MAX or deck < CARD_MIN

def easy21(state, action):
    player, dealer = state
    if action == HIT:
        player += draw_card()
        return (None, -1) if is_bust(player) else ((player, dealer), 0)
    while dealer < DEALER_STICK:
        dealer += draw_card()
        if is_bust(dealer):
            return None, 1
    return None, int(dealer < player) - int(dealer > player)

def mse(Q_a, Q_b):
    err = 0
    n = 0
    for player in range(CARD_MIN, DECK_MAX + 1):
        for dealer in range(CARD_MIN, CARD_MAX + 1):
            state = player, dealer
            for action in HIT, STICK:
                err += (Q_a(state, action) - Q_b(state, action))**2
                n += 1
    return err/n

def epsilon_policy(actions, e):
    m = len(actions)
    best = max((q, a) for a, q in actions)[1]
    weighted = [(a, (e/m + 1 - e) if a == best else e/m) for a, q in actions]
    tip = random.random()
    mass = 0
    for a, w in weighted:
        mass += w
        if tip <= mass:
            return a
    assert False

def mc(episodes):
    Q = defaultdict(lambda:{HIT:0, STICK:0})
    N = defaultdict(lambda:defaultdict(int))
    for _ in range(episodes):
        state = draw_blacks()
        episode = []
        reward = 0
        while state is not None:
            epsilon = N0/(N0 + sum(N[state].values()))
            action = epsilon_policy(Q[state].items(), epsilon)
            episode.append((state, action))
            state, reward = easy21(state, action)
        for s, a in episode:
            N[s][a] += 1
            q = Q[s][a]
            alf = 1/N[s][a]
            Q[s][a] = q + alf*(reward - q)
    return Q

def td(episodes, lmbda, Q_ref):
    Q = defaultdict(lambda:{HIT:0, STICK:0})
    N = defaultdict(lambda:defaultdict(int))
    mses = []
    for _ in range(episodes):
        E = defaultdict(lambda:defaultdict(int))
        state = draw_blacks()
        action = epsilon_policy(Q[state].items(), 1.0)
        while state is not None:
            E[state][action] += 1
            N[state][action] += 1
            q = Q[state][action]
            state, reward = easy21(state, action)
            action = epsilon_policy(Q[state].items(), N0/(N0 + sum(N[state].values())))
            err = reward + Q[state][action] - q
            for s in E:
                for a in E[s]:
                    alf = 1/N[s][a]
                    Q[s][a] += alf*err*E[s][a]
                    E[s][a] *= lmbda
        mses.append(mse(lambda s, a: Q_ref[s][a], lambda s, a: Q[s][a]))
    return Q, mses

def feature(state, action):
    DEAL = [(1, 4), (4, 7), (7, 10)]
    PLAY = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 21)]
    ACT = [HIT, STICK]
    player, dealer = state
    def indicator(d, p, a):
        if action != a:
            return 0
        if dealer < d[0] or dealer > d[1]:
            return 0
        if player < p[0] or player > p[1]:
            return 0
        return 1
    return [indicator(d, p, a) for d in DEAL for p in PLAY for a in ACT]

def qfa(state, action, w):
    return sum(map(operator.mul, feature(state, action), w)) if state is not None else 0

def fa(episodes, lmbda, Q_ref):
    W = [0]*FEATURES
    ALF = 0.01
    ETA = 0.05
    actions = lambda state: [(HIT, qfa(state, HIT, W)), (STICK, qfa(state, STICK, W))]
    mses = []
    for _ in range(episodes):
        state = draw_blacks()
        action = epsilon_policy(actions(state), ETA)
        E = [0]*FEATURES
        while state is not None:
            X = feature(state, action)
            q = qfa(state, action, W)
            state, reward = easy21(state, action)
            action = epsilon_policy(actions(state), ETA)
            err = reward + qfa(state, action, W) - q
            E = [lmbda*e + x for x, e in zip(X, E)]
            W = [w + ALF*err*e for w, e in zip(W, E)]
        mses.append(mse(lambda s, a: Q_ref[s][a], lambda s, a: qfa(s, a, W)))
    return W, mses

def eye_candy(Q_mc, Q_td, Q_fa):
    td_lmbda = lambda t: t[1]
    td_mses = lambda t: t[0][1]

    fig = plt.figure(num="Easy21 eyecandy")
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[:, 0], projection='3d')
    ax.set_title("GLIE-MC state-value ({} episodes)".format(Q_MC_EPS))
    ax.set_xlabel("Player sum")
    ax.set_ylabel("Dealer showing")
    ax.set_zlabel("V")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    def v(x, y):
        return max((q, a) for a, q in Q_mc[(x, y)].items())[0]

    player, dealer, reward = zip(
        *((x, y, v(x, y)) for x in range(CARD_MIN, DECK_MAX + 1) for y in range(CARD_MIN, CARD_MAX + 1)))
    ax.plot_trisurf(player, dealer, reward, cmap=cm.coolwarm)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("TD(λ) ↔ GLIE-MC mean squared error ({} episodes)".format(Q_TD_EPS))
    ax.plot([td_lmbda(qs) for qs in Q_td], [td_mses(qs)[-1] for qs in Q_td], label = "TD")
    ax.plot([td_lmbda(qs) for qs in Q_fa], [td_mses(qs)[-1] for qs in Q_fa], label = "TD-FA")
    ax.set_xlabel("λ")
    ax.set_ylabel("MSE")
    plt.legend()

    ax = fig.add_subplot(gs[1, 1])
    ax.set_title("TD(λ) ↔ GLIE-MC mean squared error per episode".format(Q_TD_EPS))

    ax.plot([mse for mse in td_mses(Q_td[0])], label="TD({})".format(0))
    ax.plot([mse for mse in td_mses(Q_td[-1])], label="TD({})".format(1))
    ax.plot([mse for mse in td_mses(Q_fa[0])], label="TD-FA({})".format(0))
    ax.plot([mse for mse in td_mses(Q_fa[-1])], label="TD-FA({})".format(1))
    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    Q_MC_EPS = int(1e6)
    Q_TD_EPS = int(1e3)
    LAMBDAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    print("Learning MC agent...")
    Q_mc = mc(episodes=Q_MC_EPS)
    print("Learning TD(λ) agents...")
    Q_td = [(td(episodes=Q_TD_EPS, lmbda=lmbda, Q_ref=Q_mc), lmbda) for lmbda in LAMBDAS]
    print("Learning TD-FA(λ) agents...")
    Q_fa = [(fa(episodes=Q_TD_EPS, lmbda=lmbda, Q_ref=Q_mc), lmbda) for lmbda in LAMBDAS]

    eye_candy(Q_mc, Q_td, Q_fa)
