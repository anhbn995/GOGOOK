import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB

def Evaluation_Fitness(pop, features, labels, train_indices, test_indices):
    acc = np.zeros(pop.shape[0])
    _id = 0

    for curr_solution in pop:
        data = features[:, np.where(curr_solution == 1)[0]]
        data_train = data[train_indices, :]
        data_test = data[test_indices, :]

        label_train = labels[train_indices]
        label_test = labels[test_indices]

        clf = GaussianNB().fit(X=data_train, y=label_train)
        acc[_id] = clf.score(data_test, label_test)
        _id = _id + 1
    return acc

def Select(pop, acc, num_point):
    point = np.empty((num_point, pop.shape[1]))
    for idx in range(num_point):
        acc_max = np.where(acc == np.max(acc))[0][0]
        point[idx, :] = pop[acc_max, :]
        acc[acc_max] = -99999999999
    return point


def Crossover(point, size_offspring):
    offspring = np.empty(size_offspring)
    point_cross = np.uint8(size_offspring[1]/2)

    for idx in range(size_offspring[0]):
        offspring[idx, 0:point_cross] = point[idx%point.shape[0], 0:point_cross]
        offspring[idx, point_cross:] = point[(idx+1)%point.shape[0], point_cross:]
    return offspring


def Mutation(Cross, mu=2):
    idx_mutation = np.random.randint(low=0, high=Cross.shape[1], size=mu)
    for idx in range(Cross.shape[0]):
        Cross[idx, idx_mutation] = 1 - Cross[idx, idx_mutation]
    return Cross


def GA(data_inputs, data_outputs, train, test):
    num_feature_elements = data_inputs.shape[1]

    population_size = 50 # Population size.
    point_mating = 20 # Number of point inside the mating pool.
    mu = 10 # Number of elements to mutate.

    pop_shape = (population_size, num_feature_elements)
    new_population = np.random.randint(low=0, high=2, size=pop_shape)

    best_output = []
    num_generations = 50
    for _ in range(num_generations):
        acc = Evaluation_Fitness(new_population, data_inputs, data_outputs, train, test)

        best_output.append(np.max(acc))

        point = Select(new_population, acc, point_mating)
        Cross = Crossover(point, offspring_size=(pop_shape[0]-point.shape[0], num_feature_elements))
        offspring_mutation = Mutation(Cross, mu=mu)

        new_population[0:point.shape[0], :] = point
        new_population[point.shape[0]:, :] = offspring_mutation
    return best_output, new_population

if __name__ == "__main__":

    dfData = pd.read_csv('/Users/LENOVO/OneDrive/Máy tính/Soil.csv', sep=',')

    allClasses = dfData['Y_SM']
    allFeatures = dfData.drop(['Y_SM'], axis=1)

    scaler = StandardScaler()
    allFeatures[allFeatures.columns] = scaler.fit_transform(allFeatures[allFeatures.columns])

    data_inputs = allFeatures.to_numpy()
    data_outputs = allClasses.to_numpy()

    num_sample = np.arange(data_inputs.shape[0])
    np.random.shuffle(num_sample)

    indices_train = num_sample[:int(0.6*len(num_sample))]
    indices_test = num_sample[int(0.6*len(num_sample)):int(0.8*len(num_sample))]
    indices_val = num_sample[int(0.8*len(num_sample)):]

    # indices_train = np.array([ 8, 13, 22, 14, 18, 20, 10, 27, 24,  1, 19, 38, 39, 21, 30, 35,  3, 36, 32, 25,  9, 37, 26, 16])
    # indices_test = np.array([28, 15,  4, 12,  7, 11, 31, 34])
    # indices_val = np.array([ 0, 33,  2,  5, 29, 17,  6, 23])

    print("Number of train samples: ", indices_train.shape[0])
    print("Number of test samples: ", indices_test.shape[0])
    print("Number of val samples: ", indices_val.shape[0])
    print()
    print("==== BEFORE using ALGORITHM GENERATION ====")
    data_train = data_inputs[indices_train, :]
    data_test = data_inputs[indices_test, :]
    data_val = data_inputs[indices_val, :]

    label_train = data_outputs[indices_train]
    label_test = data_outputs[indices_test]
    label_val = data_outputs[indices_val]

    clf = GaussianNB().fit(X=data_train, y=label_train)
    acc_test =  clf.score(data_test, label_test)
    print("acc train:", clf.score(data_train, label_train))
    print("acc test:", acc_test)
    print("acc val:", clf.score(data_val, label_val))
    print()
    print("Start GA....")
    print()
    best_outputs, new_population = GA(data_inputs, data_outputs, indices_train, indices_test)
    best_outputs.insert(0, acc_test)
    acc = Evaluation_Fitness(new_population, data_inputs, data_outputs, indices_train, indices_test)
    best_match_idx =  np.where(acc == np.max(acc))[0][0]
    best_solution = new_population[best_match_idx, :]
    best_outputs.append(acc[best_match_idx])
    best_solution_indices = np.where(best_solution == 1)[0]
    print("==== AFTER using ALGORITHM GENERATION ====")
    print()
    data_input = data_inputs[:, np.where(best_solution == 1)[0]]
    data_train = data_input[indices_train, :]
    data_test = data_input[indices_test, :]
    data_val = data_input[indices_val, :]

    label_train = data_outputs[indices_train]
    label_test = data_outputs[indices_test]
    label_val = data_outputs[indices_val]

    clf = GaussianNB().fit(X=data_train, y=label_train)
    print("acc train:", clf.score(data_train, label_train))
    print("acc test:", clf.score(data_test, label_test))
    print("acc val:", clf.score(data_val, label_val))
    print('-----------------')
    print('Feature is retained:',allFeatures.columns[best_solution_indices])
    plt.plot(best_outputs)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.show()