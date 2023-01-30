import numpy as np


def RootsWeightsA(rank):
    epsilon = np.identity(rank + 1)
    roots = [[epsilon[i] - epsilon[i + 1]] for i in range(len(epsilon) - 1)]
    roots = np.concatenate(roots, axis=0)
    weights_1 = []
    for i in range(1, len(epsilon)):
        weights_1.append([epsilon[j] - epsilon[i] for j in range(i)])
    weights_1 = np.concatenate(weights_1, axis=0)
    weights_2 = [sum(epsilon[j] for j in range(i)) - i / (rank + 1) * sum(epsilon) for i in range(1, rank + 1)]
    weights_3 = np.hstack([1, np.zeros(rank - 1), -1])
    return roots, weights_1, weights_2, weights_3


def RootsWeightsB(rank):
    epsilon = np.identity(rank)
    roots = np.concatenate(([epsilon[i] - epsilon[i + 1] for i in range(len(epsilon) - 1)], [epsilon[-1]]))
    weights_1 = np.concatenate(
        (
            [epsilon[j] + epsilon[i] for i in range(1, len(epsilon)) for j in range(i)],
            [epsilon[j] - epsilon[i] for i in range(1, len(epsilon)) for j in range(i)],
        )
    )
    weights_1 = np.concatenate((weights_1, epsilon))
    weights_2 = np.concatenate(([sum(epsilon[j] for j in range(i)) for i in range(1, rank)], [0.5 * sum(epsilon)]))
    weights_3 = np.concatenate(([1, 1], np.zeros(rank - 2)))
    return roots, weights_1, weights_2, weights_3


def RootsWeightsC(rank):
    epsilon = np.identity(rank)
    roots = np.concatenate(([epsilon[i] - epsilon[i + 1] for i in range(len(epsilon) - 1)], [2 * epsilon[-1]]))
    weights_1 = np.concatenate(
        (
            [epsilon[j] + epsilon[i] for i in range(1, len(epsilon)) for j in range(i)],
            [epsilon[j] - epsilon[i] for i in range(1, len(epsilon)) for j in range(i)],
        )
    )
    weights_2 = np.concatenate((weights_1, 2 * epsilon))
    weights_3 = [sum(epsilon[j] for j in range(i)) for i in range(1, rank + 1)]
    weights_4 = np.concatenate(([2], np.zeros(rank - 1)))
    return roots, weights_2, weights_3, weights_4


def RootsWeightsD(rank):
    epsilon = np.identity(rank)
    roots = np.concatenate(([epsilon[i] - epsilon[i + 1] for i in range(len(epsilon) - 1)], [epsilon[-1] + epsilon[-2]]))
    weights_1 = np.concatenate(
        (
            [epsilon[i] + epsilon[j] for i in range(1, len(epsilon)) for j in range(i)],
            [epsilon[j] - epsilon[i] for i in range(1, len(epsilon)) for j in range(i)],
        )
    )
    weight_sum = [sum(epsilon[j] for j in range(i)) for i in range(1, rank - 1)]
    if len(weight_sum) == 0:
        weights_2 = np.concatenate(([0.5 * sum(epsilon) - epsilon[rank - 1]], [0.5 * sum(epsilon)]))
    else:
        weights_2 = np.concatenate((weight_sum, [0.5 * sum(epsilon) - epsilon[rank - 1]], weight_sum, [0.5 * sum(epsilon)]))
    weights_3 = np.concatenate(([1, 1], np.zeros(rank - 2)))
    return roots, weights_1, weights_2, weights_3


def simple_roots(group, rank):
    if group == "A":
        rootWeight = RootsWeightsA(rank)
    elif group == "B":
        rootWeight = RootsWeightsB(rank)
    elif group == "C":
        rootWeight = RootsWeightsC(rank)
    elif group == "D":
        rootWeight = RootsWeightsD(rank)
    tsDim = len(rootWeight[-1])
    SimpleRoots, PositiveRoots, FundamentalWeights, MaximalRoots = rootWeight
    WeylVector = sum(PositiveRoots) / 2
    CoxeterNumber = 2 * len(PositiveRoots) / rank
    return {
        "tsDim": tsDim,
        "SimpleRoots": SimpleRoots,
        "PositiveRoots": PositiveRoots,
        "FundamentalWeights": FundamentalWeights,
        "MaximalRoots": MaximalRoots,
        "WeylVector": WeylVector,
        "CoxeterNumber": CoxeterNumber,
    }

