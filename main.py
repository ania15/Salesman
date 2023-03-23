import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import rankdata
import time


def selection_delete_and_duplicate(costs, perms, n_parents, percentage):
    """
    Selects parents from population by deleting the worst individuals and duplicating the best ones.
    Note: this function might be unsafe ;)

    :param costs: vector of cost function values for the population
    :param perms: permutations for each individual
    :param n_parents: number of individuals in each generation
    :param percentage: percentage of worst individuals to be destroyed
    :returns: new parents and their new costs
    """
    # Sorting costs (indices)
    idx = np.argsort(costs)

    # sorting population according to the cost function
    parents = perms[:, idx[:n_parents]]

    # counting number of individuals to be destroyed
    n_destroy = int(percentage * n_parents)

    # duplicating
    best_parents = np.zeros(perms.shape)
    best_costs = np.zeros(costs.shape)
    best_parents[:, :(n_parents-n_destroy)] = parents[:, :(n_parents-n_destroy)]
    best_costs[:(n_parents-n_destroy)] = costs[idx[:n_parents - n_destroy]]
    rest_parents_idx = np.random.choice(np.arange(n_parents - n_destroy), size=n_destroy, replace=True)
    best_parents[:, (n_parents-n_destroy):] = best_parents[:, rest_parents_idx]
    best_costs[(n_parents-n_destroy):] = costs[idx[rest_parents_idx]]

    return best_parents, best_costs


def rank_sel(costs, n_parents):
    """
    Function performing selection of individuals by rank method.

    :param n_parents: number of parents to select
    :param costs: vector of cost function values for the population
    :return: indexes of parents chosen based on the rankings
    """
    # Compute the ranks of all the costs and their sum
    ranks = rankdata(costs, method='ordinal')
    rank_sum = sum(ranks)

    # Compute the probability of selection for each cost and cumulative sum of the probabilities
    probabilities = ranks / rank_sum
    cum_probabilities = [sum(probabilities[:i + 1]) for i in range(len(probabilities))]

    # Generate n_parents random numbers between 0 and 1
    random_numbers = [random.random() for _ in range(n_parents)]

    # Select the parents
    parents_indices = []
    for random_number in random_numbers:
        for i in range(len(cum_probabilities)):
            if cum_probabilities[i] >= random_number:
                parents_indices.append(i)
                break
    return parents_indices


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


def crossing(perms, parents_idx, prob):
    """
    Function implementing crossing of genes between individuals

    :param prob: probability of crossing
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
    min_cross = int(n_cities * 0.2)
    max_cross = int(n_cities * 0.8)
    n_parents = int(n_populations / 2)  # number of pairs of parents f.e. 6 pairs = 12 parents = 12 children

    # creating empty array for next generation
    children = np.zeros((n_cities, n_populations))

    # crossing with probability

    for pair_id in range(n_parents):
        if np.random.rand() < prob:
            # drawing random places where crossing will take place
            length_of_cross = np.random.randint(min_cross, max_cross)  # 20-80% of length of permutation will be changed
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
        else:
            children[:, pair_id * 2] = parents[:, pair_id * 2]
            children[:, pair_id * 2 + 1] = parents[:, pair_id * 2 + 1]
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


def salesman_gen(num_cities=10, n_population=100, mutation_prob=0.1, cross_prob=0.9,
                 n_changes=1000, percentage_bad=0.2, selection_method='roulette'):
    """
    Function solving Traveling Salesman problem using genetic algorithm

    :param cross_prob: probability of crossing
    :param selection_method: Can be 'roulette' or 'rank'
    :param mutation_prob: probability of mutation
    :param num_cities: number of cities we want to travel to
    :param n_population: number of population for our algorithm
    :param percentage_bad: percentage of bad individuals to kill
    :param n_changes: number of generations without change to finish looking for solution
    :return: best route, x and y coordinates, list of best results
    """

    # starting measuring time
    start = time.time()

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
    best_list = []

    # variables for controlling number of generations
    no_change = 0
    i = 0

    # main algorithm
    while no_change < n_changes:

        # grading current generation
        costs, best = euclidean_sum(x, y, perms)

        # printing current best result for debugging
        #print("Generacja:", i, "total:", total_best_cost, "populacja:", best)

        # if we got better permutation - save it
        if best < total_best_cost:
            total_best_cost = best
            best_solution = perms[:, np.argmin(costs)]
            no_change = 0
        else:
            no_change += 1

        # appending best cost to list for visualization
        best_list.append(total_best_cost)

        # duplicating good permutations and deleting bad ones
        perms, new_costs = selection_delete_and_duplicate(costs, perms, n_population, percentage_bad)

        # selection
        parents_idx = selection(new_costs, n_population)

        # crossing
        children = crossing(perms, parents_idx, cross_prob)

        # mutating
        children_mutated = mutation(children, mutation_prob)

        # saving new generation
        perms = children_mutated

        # incrementing number of generations
        i += 1

    # printing final best solution
    end = time.time()
    best_solution = best_solution.astype(int)
    print("Najlepsze rozwiązanie:", best_solution, "o koszcie:", total_best_cost,
          "\nZnalezione w", i, "iteracjach, w czasie: ", end - start, "sekund.")

    return best_solution, x, y, best_list


def visualize(best_solution_ra, x_ra, y_ra, best_ra, best_solution_ro, x_ro, y_ro, best_ro):
    """

    :param best_solution_ra: best route (number of cities in order) - rank selection
    :param x_ra: x coordinates of the cities - rank selection
    :param y_ra: y coordinates of the cities - rank selection
    :param best_ra: list of the bets costs in each population - rank selection
    :param best_solution_ro: best route (number of cities in order) - roulette selection
    :param x_ro: x coordinates of the cities - roulette selection
    :param y_ro: x coordinates of the cities - roulette selection
    :param best_ro: list of the bets costs in each population - roulette selection
    """
    fig, ax = plt.subplots(2, 2, figsize=[15, 15])

    fig.suptitle("Problem Komiwojażera")
    ax[0][0].set(title="Optymalna ścieżka (selekcja metodą rankingową)", xlabel="X", ylabel="Y")
    ax[0][0].scatter(x_ra, y_ra, color='b')
    for i in range(len(best_solution_ra) - 1):
        start = best_solution_ra[i]
        end = best_solution_ra[i + 1]
        x_vals = [x_ra[start], x_ra[end]]
        y_vals = [y_ra[start], y_ra[end]]
        ax[0][0].plot(x_vals, y_vals, color='r')
        ax[0][0].text(x_ra[start], y_ra[start], str(start), fontsize=12)
    # pierwsze z ostatnim
    x_vals = [x_ra[best_solution_ra[-1]], x_ra[best_solution_ra[0]]]
    y_vals = [y_ra[best_solution_ra[-1]], y_ra[best_solution_ra[0]]]
    ax[0][0].plot(x_vals, y_vals, color='r')
    ax[0][0].text(x_ra[best_solution_ra[-1]], y_ra[best_solution_ra[-1]], str(best_solution_ra[-1]))

    ax[0][1].set(title="Funkcja kosztu w kolejnych pokoleniach", xlabel="Numer pokolenia",
                 ylabel="Wartość funkcji kosztu")
    ax[0][1].plot(best_ra)

    ax[1][0].set(title="Optymalna ścieżka (selekcja metodą ruletki)", xlabel="X", ylabel="Y")
    ax[1][0].scatter(x_ro, y_ro, color='b')
    for i in range(len(best_solution_ro) - 1):
        start = best_solution_ro[i]
        end = best_solution_ro[i + 1]
        x_vals = [x_ro[start], x_ro[end]]
        y_vals = [y_ro[start], y_ro[end]]
        ax[1][0].plot(x_vals, y_vals, color='r')
        ax[1][0].text(x_ro[start], y_ro[start], str(start), fontsize=12)
    # pierwsze z ostatnim
    x_vals = [x_ro[best_solution_ro[-1]], x_ro[best_solution_ro[0]]]
    y_vals = [y_ro[best_solution_ro[-1]], y_ro[best_solution_ro[0]]]
    ax[1][0].plot(x_vals, y_vals, color='r')
    ax[1][0].text(x_ro[best_solution_ro[-1]], y_ro[best_solution_ro[-1]], str(best_solution_ro[-1]))

    ax[1][1].set(title="Funkcja kosztu w kolejnych pokoleniach", xlabel="Numer pokolenia",
                 ylabel="Wartość funkcji kosztu")
    ax[1][1].plot(best_ro)

    plt.show()


if __name__ == '__main__':
    # np.random.seed(42)
    best_sol_ra, x_ra, y_ra, best_ra = salesman_gen(num_cities=20,
                                                    n_population=200,
                                                    n_changes=500,
                                                    selection_method='rank')

    best_sol_ro, x2_ro, y2_ro, best_ro = salesman_gen(num_cities=20,
                                                      n_population=200,
                                                      n_changes=500,
                                                      selection_method='roulette')

    visualize(best_sol_ra, x_ra, y_ra, best_ra, best_sol_ro, x2_ro, y2_ro, best_ro)
