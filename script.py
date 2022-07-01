import random
from random import randint
import numpy as  np
import numpy.random as npr
import matplotlib.pyplot as plt

def objectiveFunction(solution):
    # our fitness function f(x) = x1^2 + x2^2 + x3 / 2 - x4 * x5
    return solution[0] ** 2 + solution[1] ** 2 + solution[2] / 2 - solution[3] * solution[4]

def selectOneWithWheel(population, fitValues):
    # normalizing fitValues since fitness function can be negative
    minValue = min(fitValues)
    fitValues = np.asarray(fitValues) + abs(minValue)
    max = sum(fitValues)
    selection_probs = [abs(value/max) for value in fitValues]
    print(selection_probs, sum(selection_probs))
    labels = list()
    for i in range(0, 8):
        label = 'Solution ' + str(i)
        labels.append(label)
    print(selection_probs)
    plt.pie(selection_probs, labels=labels)
    plt.show()
    return population[npr.choice(len(population), p=selection_probs)]

def crossoverBreed(s1, s2):
    c1, c2 = s1.copy(), s2.copy()
    # crossover happens with 70% probability

    if random.random() < 0.7:
        # select crossover point with at least 2 genes dimension
        pt = randint(1, len(s1)-2)
        # perform crossover
        c1 = s1[:pt] + s2[pt:]
        c2 = s2[:pt] + s1[pt:]
    return [c1, c2]

def randomMutation(solution):
    for i in range(len(solution)):
        # mutation happens with a 10% chance
        if random.random() < 0.1:
            # sets bit to the opposite value
            solution[i] = 1 - solution[i]


# setting up the initial population, made up of 8 elements
# each of them represented by a sequence 5 bits
pop = [npr.randint(0, 2, 5).tolist() for _ in range(8)]
best, best_eval = 0, objectiveFunction(pop[0])

# defining 200 iterations for our genic algorithm
for generation in range(200):

    #calculates fitness values for each solution
    fitnessValues = [objectiveFunction(solution) for solution in pop]

    #check if the previously calculated fitness values are better than the 'best' currently stored
    for i in range(8):
        if fitnessValues[i] < best_eval:
            best, best_eval = pop[i], fitnessValues[i]

    # selects solutions for breeding
    selectedSolutions = [selectOneWithWheel(pop, fitnessValues) for _ in range(8)]

    # declare new list of children solutions, currently empty
    children = list()
    for i in range(0, 8, 2):
        # get couples of selected parents
        parent1, parent2 = selectedSolutions[i], selectedSolutions[i+1]
        # applies crossover and mutation
        for c in crossoverBreed(parent1, parent2):
            randomMutation(c)
            # saves newly generated children
            children.append(c)
    #replaces all items of previous generations with new ones
    pop = children

print(best, best_eval)