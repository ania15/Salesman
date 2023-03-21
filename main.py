import numpy as np
import matplotlib.pyplot as plt


def selection_delete_and_duplicate(costs, perms, n_parents):
    """
    Selects parents from population by deleting the worst individuals and duplicating the best ones.
    Note: this function might be unsafe ;)

    :param costs: vector of cost function values for the population
    :param perms: permutations for each individual
    :param n_parents: number of individuals in each generation
    """
    # Sorting costs (indices)
    idx = np.argsort(costs)

    # sorting population according to the cost function
    parents = perms[:, idx[:n_parents]]

    # duplicating first half
    good_parents = np.repeat(parents[:, :n_parents // 2], 2, axis=1)
    new_costs = np.repeat(costs[idx[:n_parents // 2]], 2)

    return good_parents, new_costs


def rank_sel(costs, n_parents):
    """
    Function performing selection of individuals by rank method.

    :param n_parents: number of parents to select
    :param costs: vector of cost function values for the population
    :return: indexes of parents chosen based on the rankings
    """
    # getting indices of best individuals
    ranks = np.argsort(costs)

    # ranking
    p = 0.4
    probs = np.zeros(len(ranks))
    total = 0
    for i in ranks[:-1]:
        probs[i] = (1 - total) * p
        total += probs[i]
    probs[ranks[-1]] = 1 - total

    # choosing parents - each 2 indices make one couple
    parents_idx = np.random.choice(np.arange(len(costs)), size=n_parents, replace=True, p=probs)

    return parents_idx


def roulette_sel(costs, n_parents):
    """
    Function performing selection of individuals by roulette method.

    :param n_parents: number of parents to select
    :param costs: vector of cost function values for the population
    :return: indexes of parents chosen based on the rankings
    """
    # getting probabilities
    prob = np.min(costs) / costs
    prob = prob / np.sum(prob)

    # choosing parents - each 2 indices make one couple
    parents_idx = np.random.choice(np.arange(len(prob)), size=n_parents, replace=True, p=prob)

    return parents_idx


def mutation(perms, prob):
    """
    Function performing mutations

    :param perms: array with permutations of cities for each individual in population
    :param prob: probability of mutation
    :return: mutated population
    """

    # copying permutations
    mutated = perms.copy()

    # getting number of cities and number of individuals
    n_cities, n_populations = mutated.shape

    # performing mutations with given probability
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

    # sorting permutations in descending order by cost function
    parents = perms[:, parents_idx]

    # getting number of cities and individuals
    n_cities, n_populations = perms.shape

    # getting length of crossing and number of couples
    length_of_cross = int(n_cities / 3)  # a piece of approximately 30% of length of permutation will be changed
    n_parents = int(n_populations / 2)  # number of pairs of parents f.e. 6 pairs = 12 parents = 12 children

    # creating empty array for next generation
    children = np.zeros((n_cities, n_populations))

    # crossing
    for pair_id in range(n_parents):
        # drawing random places where crossing will take place
        start = np.random.randint(0, n_cities - length_of_cross)
        stop = start + length_of_cross

        # selecting parents
        p1 = parents[:, pair_id * 2]
        p2 = parents[:, pair_id * 2 + 1]

        # selecting parts which will be changed
        cities_1 = p1[start:stop]
        cities_2 = p2[start:stop]

        # getting indexes of cities we want to change
        index_1 = np.ravel([np.where(p1 == i) for i in cities_2])
        index_2 = np.ravel([np.where(p2 == i) for i in cities_1])

        # deleting those cities
        p1_new = np.delete(p1, index_1)
        p2_new = np.delete(p2, index_2)

        # inserting new cities
        p1_final = np.insert(p1_new, start, cities_2)
        p2_final = np.insert(p2_new, start, cities_1)

        # putting it to final array
        children[:, pair_id * 2] = p1_final
        children[:, pair_id * 2 + 1] = p2_final

    return children


def euclidean_sum(x, y, perms):
    """
    Cost function using Euclidean norm. Sums all the distances between each of the cities.

    :param x: x coordinates of cities
    :param y: y coordinates of cities
    :param perms: array with shape (n_cities, n_populations) with permutations of cities for each
    individual in population
    :return: sum of distances for each individual and result of best individual
    """
    # shifting permutations so that first city is last
    perms_shifted = np.roll(perms, -1, axis=0)

    # getting coordinates of shifted permutations
    x_moved = x[perms_shifted.astype(int).T]
    y_moved = y[perms_shifted.astype(int).T]

    # getting coordinates of not shifted permutations
    x_new = x[perms.astype(int).T]
    y_new = y[perms.astype(int).T]

    # counting distances
    dist = ((x_moved - x_new) ** 2 + (y_moved - y_new) ** 2) ** 0.5

    return np.sum(dist, axis=1), np.min(np.sum(dist, axis=1))


def salesman_gen(num_cities=10, n_population=100, n_generations=100, mutation_prob=0.5,
                 selection_method='roulette'):
    """
    Function solving Traveling Salesman problem using genetic algorithm

    :param selection_method: Can be 'roulette' or 'rank'
    :param mutation_prob: probability of mutation
    :param num_cities: number of cities we want to travel to
    :param n_population: number of population for our algorithm
    :param n_generations: number of generations to test
    :return:
    """
    # drawing x and y coordinates of cities
    x = 300 * np.random.random(num_cities)
    y = 300 * np.random.random(num_cities)

    # generating initial population
    perms = np.zeros((num_cities, n_population))

    for i in range(n_population):
        perms[1:, i] = np.random.permutation(num_cities - 1) + 1

    # deciding method of selection
    if selection_method == 'roulette':
        selection = roulette_sel
    elif selection_method == 'rank':
        selection = rank_sel
    else:
        print('This selection method is not supported')
        exit(2)

    # grading initial population
    costs, total_best_cost = euclidean_sum(x, y, perms)
    best_solution = perms[:, np.argmin(costs)]

    # list for visualization
    best_list = np.zeros((n_generations, 1))

    # main algorithm
    for i in range(n_generations):

        # grading current generation
        costs, best = euclidean_sum(x, y, perms)

        # printing current best result for debugging
        print("Generacja:", i, "total:", total_best_cost, "populacja:", best)

        # if we got better permutation - save it
        if best < total_best_cost:
            total_best_cost = best
            best_solution = perms[:, np.argmin(costs)]

        # appending best cost to list for visualization
        best_list[i] = total_best_cost

        # duplicating good permutations and deleting bad ones
        perms, new_costs = selection_delete_and_duplicate(costs, perms, n_population)

        # selection
        parents_idx = selection(new_costs, n_population)

        # crossing
        children = crossing(perms, parents_idx)

        # mutating
        children_mutated = mutation(children, mutation_prob)

        # saving new generation
        perms = children_mutated

    # printing final best solution
    best_solution = best_solution.astype(int)
    print("Najlepsze rozwiązanie:", best_solution, "o koszcie:", total_best_cost)

    return best_solution, x, y, best_list


def visualize(best_solution, x, y, best, sel):
    fig, ax = plt.subplots(1, 2, figsize=[10, 5])
    if sel == 'rank':
        fig.suptitle("Problem Komiwojażera (selekcja rankingowa)")
    elif sel == 'roulette':
        fig.suptitle("Problem Komiwojażera (selekcja ruletką)")

    ax[0].set(title="Optymalna ścieżka", xlabel="X", ylabel="Y")
    ax[0].scatter(x, y, color='b')
    for i in range(len(best_solution) - 1):
        start = best_solution[i]
        end = best_solution[i + 1]
        x_vals = [x[start], x[end]]
        y_vals = [y[start], y[end]]
        ax[0].plot(x_vals, y_vals, color='r')
        ax[0].text(x[start], y[start], str(start), fontsize=12)
    # pierwsze z ostatnim
    x_vals = [x[best_solution[-1]], x[best_solution[0]]]
    y_vals = [y[best_solution[-1]], y[best_solution[0]]]
    ax[0].plot(x_vals, y_vals, color='r')
    ax[0].text(x[best_solution[-1]], y[best_solution[-1]], str(best_solution[-1]))

    ax[1].set(title="Funkcja kosztu w kolejnych pokoleniach", xlabel="Numer pokolenia", ylabel="Wartość funkcji kosztu")
    ax[1].plot(best)

    plt.show()


if __name__ == '__main__':
    # np.random.seed(42)
    best_sol, x, y, best = salesman_gen(num_cities=10,
                                        n_population=300,
                                        n_generations=200,
                                        mutation_prob=0,
                                        selection_method='rank')

    visualize(best_sol, x, y, best, 'rank')

    exit(0)
