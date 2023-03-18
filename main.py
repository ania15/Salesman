import numpy as np


def rank_sel(costs, n_parents):
    """
    :param n_parents: number of parents to select
    :param costs: vector of cost function values for the population
    :return: indexes of parents chosen based on the rankings
    """

    ranks = np.argsort(costs)
    ranks = np.sort(ranks)

    p = 0.67
    probs = np.zeros(len(ranks))
    for i in range(len(ranks) - 1):
        probs[i] = (1 - sum(probs[:i])) * p
    probs[-1] = 1 - sum(probs)

    parents_idx = np.random.choice(np.arange(len(costs)), size=n_parents, replace=False, p=probs)
    return parents_idx


def roulette_sel(costs, n_parents):
    """
    :param n_parents: number of parents to select
    :param costs: vector of cost function values for the population
    :return: indexes of parents chosen based on the rankings
    """

    prob = np.min(costs) / costs
    prob = prob / np.sum(prob)

    parents_idx = np.random.choice(np.arange(len(prob)), size=n_parents, replace=False, p=prob)

    return parents_idx


def mutation(perms, prob):
    """
    :param perms: array with permutations of cities for each individual in population
    :param prob: probability of mutation
    :return: mutated population
    """
    mutated = perms.copy()
    n_cities, n_populations = mutated.shape
    for i in range(n_populations):
        if np.random.rand() < prob:
            city1, city2 = np.random.choice(np.arange(1, n_cities), size=2, replace=False)
            mutated[city1, i], mutated[city2, i] = mutated[city2, i], mutated[city1, i]

    return mutated


def crossing(perms, parents_idx):
    """
    Function implementing crossing of genes between individuals

    :param parents_idx: indexes of parents, every 2 following indexes make parents
    :param perms: array with shape (n_cities, n_populations) of permutations, each column index is another individual,
        later becoming a child
    :return: children, got by crossing genes of parents
    """
    parents = perms[:, parents_idx]
    n_cities, n_populations = perms.shape

    length_of_cross = int(n_cities / 3)  # a piece of approximately 30% of length of permutation will be changed
    n_parents = int(n_populations / 2)  # number of pairs of parents f.e. 6 pairs = 12 parents = 12 children

    for pair_id in range(n_parents):
        start = np.random.randint(0, n_cities - length_of_cross)  # można wyciągnąć przed pętle i zoptymalizowac
        stop = start + length_of_cross  # można wyciągnąć przed pętle i zoptymalizowac

        p1 = parents[:, pair_id * 2]
        p2 = parents[:, pair_id * 2 + 1]
        cities_1 = p1[start:stop]
        cities_2 = p2[start:stop]

        for el in range(len(p1)): # tu dalej
            print(el)

    return parents


def euclidean_sum(x, y, perms):
    """
    Cost function using Euclidean norm. Sums all the distances between each of the cities.

    :param x: x coordinates of cities
    :param y: y coordinates of cities
    :param perms: array with shape (n_cities, n_populations) with permutations of cities for each
    individual in population
    :return: sum of distances for each individual and result of best individual
    """
    perms_shifted = np.roll(perms, -1, axis=0)

    x_moved = x[perms_shifted.astype(int).T]
    y_moved = y[perms_shifted.astype(int).T]

    x_new = x[perms.astype(int).T]
    y_new = y[perms.astype(int).T]

    dist = ((x_moved - x_new) ** 2 + (y_moved - y_new) ** 2) ** 0.5

    return np.sum(dist, axis=1), np.min(np.sum(dist, axis=1))


def salesman_gen(num_cities, n_population, n_generations, mutation_prob):
    """
    Function solving Traveling Salesman problem using genetic algorithm

    :param mutation_prob:
    :param num_cities:
    :param n_population:
    :param n_generations:
    :return:
    """
    x = 300 * np.random.random(num_cities)
    y = 300 * np.random.random(num_cities)

    perms = np.zeros((num_cities, n_population))

    for i in range(n_population):
        perms[1:, i] = np.random.permutation(num_cities - 1) + 1

    for i in range(n_generations):
        costs, best = euclidean_sum(x, y, perms)
        parents_idx = roulette_sel(costs, n_population)  # rank_sel(costs, n_parents)

        children = crossing(perms, parents_idx)
        children_mutated = mutation(children, mutation_prob)
        perms = children_mutated


if __name__ == '__main__':
    salesman_gen(10, 12, 1, 0.1)
