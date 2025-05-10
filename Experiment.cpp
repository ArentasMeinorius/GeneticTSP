// Experiment.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <map>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <set>


using namespace std;

string TEST_INPUT_LOCATION = "Inputs\\";
string OUTPUT_FILE = "results.csv";

int POPULATION_SIZE = 100;
int GENERATIONS = 5;
int TOKENVALUE = 50;
int TRIES = 1;
int MUTATION_PERCENTAGE = 1;
random_device my_random_device;
mt19937 generator(my_random_device());
vector<int> operatorSequence;
int cityCount;


typedef pair<vector<int>, double> citizen;
typedef pair<double, double> coordinates;


#pragma region ExperimentFunctions

vector<int> createInitialPath(const vector<vector<double>>& distance_matrix) {
    vector<int> path(cityCount);
    iota(path.begin(), path.end(), 0);

    shuffle(path.begin() + 1, path.end(), generator);
    return path;
}

double calculateCost(const vector<int>& path, const vector<vector<double>>& distance_matrix) {
    double cost = 0;
    for (int i = 1; i < path.size(); i++) {
        cost += distance_matrix[path[i - 1]][path[i]];
    }
    cost += distance_matrix[path.back()][0]; // close circle
    return cost;
}

citizen makeCitizen(const vector<int>& path, const vector<vector<double>>& distance_matrix) {
    auto distance = calculateCost(path, distance_matrix);
    return citizen(path, distance);
}

vector<int> indexOperation(const vector<int>& list1, const vector<int>& list2, int index) {
    vector<int> combinedList;
    set<int> visited;

    for (int i = 0; i < index; i++) {
        combinedList.push_back(list1[i]);
        visited.insert(list1[i]);
    }
    for (int i = 0; i < list2.size(); i++) {
        if (visited.find(list2[i]) == visited.end()) {
            combinedList.push_back(list2[i]);
        }
    }
    return combinedList;
}

citizen crossoverOrder(const citizen& parentA, const citizen& parentB, const vector<vector<double>>& distance_matrix) {
    uniform_int_distribution<> dis(0, cityCount);
    auto crossoverIndex = dis(generator);
    auto newPath = indexOperation(parentA.first, parentB.first, crossoverIndex);
    return makeCitizen(newPath, distance_matrix);
}

citizen crossoverGreedy(const citizen& parent1, const citizen& parent2, const vector<vector<double>>& distance_matrix) {
    vector<int> offspring;
    set<int> visited;

    auto current_city = parent1.first[0];
    offspring.push_back(current_city);
    visited.insert(current_city);

    for (int i = 1; i < cityCount; ++i) {
        int next_city = -1;
        double min_distance = numeric_limits<double>::max();

        int candidate_city1 = parent1.first[(find(parent1.first.begin(), parent1.first.end(), current_city) - parent1.first.begin() + 1) % cityCount];
        int candidate_city2 = parent2.first[(find(parent2.first.begin(), parent2.first.end(), current_city) - parent2.first.begin() + 1) % cityCount];

        if (visited.find(candidate_city1) == visited.end() && distance_matrix[current_city][candidate_city1] < min_distance) {
            min_distance = distance_matrix[current_city][candidate_city1];
            next_city = candidate_city1;
        }

        if (visited.find(candidate_city2) == visited.end() && distance_matrix[current_city][candidate_city2] < min_distance) {
            min_distance = distance_matrix[current_city][candidate_city2];
            next_city = candidate_city2;
        }

        if (next_city == -1) {
            for (int k = 0; k < cityCount; ++k) {
                if (visited.find(k) == visited.end() && distance_matrix[current_city][k] < min_distance) {
                    min_distance = distance_matrix[current_city][k];
                    next_city = k;
                }
            }
        }

        offspring.push_back(next_city);
        visited.insert(next_city);
        current_city = next_city;
    }
    return makeCitizen(offspring, distance_matrix);
}

list<int> buildEdgeList(const vector<int>& parent, int city) {
    list<int> edges;
    auto it = find(parent.begin(), parent.end(), city);
    if (it != parent.end()) {
        if (it != parent.begin()) {
            edges.push_back(*(it - 1));
        }
        if (it + 1 != parent.end()) {
            edges.push_back(*(it + 1));
        }
        else {
            edges.push_back(parent.front()); //full circle
        }
    }
    return edges;
}

citizen crossoverEdgeRecombination(const citizen& parent1, const citizen& parent2, const vector<vector<double>>& distance_matrix) {
    vector<int> offspring(cityCount);
    set<int> visited;
    unordered_map<int, set<int>> edgeMap;

    for (int city : parent1.first) {
        auto edges = buildEdgeList(parent1.first, city);
        edgeMap[city].insert(edges.begin(), edges.end());
        edges = buildEdgeList(parent2.first, city);
        edgeMap[city].insert(edges.begin(), edges.end());
    }

    int current_city = parent1.first[0];
    offspring[0] = current_city;
    visited.insert(current_city);

    for (int i = 1; i < cityCount; ++i) {
        for (auto& entry : edgeMap) {
            entry.second.erase(current_city);
        }

        int next_city = -1;
        size_t fewest_edges = numeric_limits<size_t>::max();

        for (int neighbor : edgeMap[current_city]) {
            if (visited.find(neighbor) == visited.end() && edgeMap[neighbor].size() < fewest_edges) {
                fewest_edges = edgeMap[neighbor].size();
                next_city = neighbor;
            }
        }

        if (next_city == -1) {
            for (const auto& entry : edgeMap) {
                if (visited.find(entry.first) == visited.end()) {
                    next_city = entry.first;
                    break;
                }
            }
        }

        offspring[i] = next_city;
        visited.insert(next_city);
        current_city = next_city;
    }

    return makeCitizen(offspring, distance_matrix);
}

citizen crossoverCycle(const citizen& parent1, const citizen& parent2, const vector<vector<double>>& distance_matrix) {
    vector<int> offspring(cityCount, -1);
    vector<bool> visited(cityCount, false);


    int start = 0;
    int next, current;

    while (find(visited.begin(), visited.end(), false) != visited.end()) {
        current = start;
        do {
            offspring[current] = parent1.first[current];
            visited[current] = true;
            //next parent 1 index to be used in solution
            next = find(parent1.first.begin(), parent1.first.end(), parent2.first[current]) - parent1.first.begin();
            current = next;
        } while (current != start);

        auto it = find(visited.begin(), visited.end(), false);
        if (it != visited.end()) {
            start = it - visited.begin();
        }
    }

    for (int i = 0; i < cityCount; ++i) {
        if (offspring[i] == -1) {
            offspring[i] = parent2.first[i];
        }
    }

    return makeCitizen(offspring, distance_matrix);
}

discrete_distribution<> makeRoulletteWheel(const vector<citizen>& population) {
    vector<double> weights;
    double sum = 0;

    for (int i = 0; i < population.size(); i++) {
        sum += population[i].second;
    }
    for (int i = 0; i < population.size(); i++) {
        auto proportion = population[i].second / sum;
        weights.push_back(1 / proportion);
    }

    discrete_distribution<> disttribution(weights.begin(), weights.end());
    return disttribution;
}

discrete_distribution<> makeOperatorSelection(int orderWeight, int cycleWeight, int greedyWeight, int edxWeight) {
    vector<int> weights = { orderWeight, cycleWeight, greedyWeight, edxWeight };
    discrete_distribution<> distribution(weights.begin(), weights.end());
    return distribution;
}

vector<int>::iterator makeOperatorSequence(int orderWeight, int cycleWeight, int greedyWeight, int edxWeight) {
    auto total = GENERATIONS * POPULATION_SIZE;
    double weightSum = orderWeight + cycleWeight + greedyWeight + edxWeight;
    operatorSequence.clear();
    operatorSequence.insert(operatorSequence.end(), total * (orderWeight / weightSum), 0);
    operatorSequence.insert(operatorSequence.end(), total * (cycleWeight / weightSum), 1);
    operatorSequence.insert(operatorSequence.end(), total * (greedyWeight / weightSum), 2);
    operatorSequence.insert(operatorSequence.end(), total * (edxWeight / weightSum), 3);
    shuffle(operatorSequence.begin() + 1, operatorSequence.end(), generator);
    return operatorSequence.begin();
}

vector<citizen> makeNewPopulation(const vector<citizen>& oldPopulation, discrete_distribution<>& parentDistribution, vector<int>::iterator operatorSequence, const vector<vector<double>>& distance_matrix) {
    vector<citizen> newPopulation;
    while (newPopulation.size() != oldPopulation.size()) {
        auto parentA = parentDistribution(generator);
        auto parentB = parentDistribution(generator);

        citizen child;
        auto crossoverOperator = *operatorSequence;
        operatorSequence++;
        if (crossoverOperator == 0) {
            child = crossoverOrder(oldPopulation[parentA], oldPopulation[parentB], distance_matrix);
        }
        else if (crossoverOperator == 1) {
            child = crossoverCycle(oldPopulation[parentA], oldPopulation[parentB], distance_matrix);
        }
        else if (crossoverOperator == 2) {
            child = crossoverGreedy(oldPopulation[parentA], oldPopulation[parentB], distance_matrix);
        }
        else if (crossoverOperator == 3) {
            child = crossoverEdgeRecombination(oldPopulation[parentA], oldPopulation[parentB], distance_matrix);
        }
        newPopulation.push_back(child);
    }
    return newPopulation;
}

vector<citizen> combineLists(const vector<citizen>& list1, const vector<citizen>& list2) {
    vector<citizen> mergedList;
    mergedList.reserve(list1.size() + list2.size());
    mergedList.insert(mergedList.end(), list1.begin(), list1.end());
    mergedList.insert(mergedList.end(), list2.begin(), list2.end());
    return mergedList;
}

vector<citizen> mutatePopulation(vector<citizen> population, vector<vector<double>> distance_matrix) {
    uniform_int_distribution<int> distribution(1, 100);

    for (int i = 0; i < population.size(); i++) {
        auto shouldMutate = distribution(generator) <= MUTATION_PERCENTAGE;
        if (!shouldMutate) {
            continue;
        }
        shuffle(population[i].first.begin() + 1, population[i].first.end(), generator);
        population[i].second = calculateCost(population[i].first, distance_matrix);
    }
    return population;
}


bool compareCitizens(const citizen& a, const citizen& b) {
    return a.second < b.second;
}

void sortByFitness(vector<citizen>& population) {
    sort(population.begin(), population.end(), compareCitizens);
}

void keepFittest(vector<citizen>& population) {
    population.resize(POPULATION_SIZE);
}

#pragma endregion

#pragma region TestReadingFunctions

vector<coordinates> readFromListFile(string name) {
    ifstream inputFile(name);

    if (!inputFile.is_open()) {
        throw runtime_error("Could not read file");
    }

    double x, y;
    vector<coordinates> locations;
    while (inputFile >> x >> y) {
        locations.push_back(coordinates(x, y));
    }

    inputFile.close();
    return locations;
}

vector<vector<double>> readFromMatrixFile(string name) {
    ifstream inputFile(name);

    if (!inputFile.is_open()) {
        throw runtime_error("Could not read file");
    }
    vector<vector<double>> distanceMatrix;

    int size;
    inputFile >> size;
    double distance;
    for (int i = 0; i < size; i++) {
        vector<double> distances;
        for (int j = 0; j < size; j++) {
            inputFile >> distance;
            distances.push_back(distance);
        }
        distanceMatrix.push_back(distances);
    }

    return distanceMatrix;
}
//UPPER_ROW 
vector<vector<double>> readFromHalfMatrixFile(string name) {
    ifstream inputFile(name);

    if (!inputFile.is_open()) {
        throw runtime_error("Could not read file");
    }
    vector<vector<double>> distanceMatrix;

    int size;
    inputFile >> size;
    size--;
    double distance;
    for (int i = 0; i < size; i++) {
        vector<double> distances;
        distances.push_back(0);
        for (int j = i; j < size; j++) {
            inputFile >> distance;
            distances.push_back(distance);
        }
        distanceMatrix.push_back(distances);
    }
    vector<double> distances;
    distances.push_back(0);
    distanceMatrix.push_back(distances);
    size++;
    //test
    for (int i = 0; i < size; i++) {
        for (int j = i - 1; j >= 0; j--) {
            distanceMatrix[i].insert(distanceMatrix[i].begin(), distanceMatrix[j][i]);
        }
    }

    return distanceMatrix;
}

double calculateDistance(coordinates point, coordinates pointTwo) {
    auto xDiff = point.first - pointTwo.first;
    auto yDiff = point.second - pointTwo.second;
    return sqrt(xDiff * xDiff + yDiff * yDiff);
}

vector<vector<double>> conversion(vector<vector<double>> distanceMatrix) {
    return distanceMatrix;
}

vector<vector<double>> conversion(vector<coordinates> coordinates) {
    auto size = coordinates.size();
    vector<vector<double>> distanceMatrix;
    for (int i = 0; i < size; i++) {
        vector<double> distances;
        for (int j = 0; j < size; j++) {
            if (i < j) {
                distances.push_back(calculateDistance(coordinates[i], coordinates[j]));
                continue;
            }
            if (i == j) {
                distances.push_back(0);
                continue;
            }
            if (i > j) {
                distances.push_back(distanceMatrix[j][i]);
                continue;
            }
        }
        distanceMatrix.push_back(distances);
    }
    return distanceMatrix;
}

#pragma endregion


int main()
{
    cout << "Starting calculations!\n";

    auto input = readFromListFile(TEST_INPUT_LOCATION + "input42.txt");
    //auto input = readFromListFile(TEST_INPUT_LOCATION + "input48.txt");
    //auto input = readFromListFile(TEST_INPUT_LOCATION + "input52.txt");
    //auto input = readFromHalfMatrixFile(TEST_INPUT_LOCATION + "input58.txt");
    //auto input = readFromListFile(TEST_INPUT_LOCATION + "input76.txt");
    auto distance_matrix = conversion(input);

    ofstream outputFile(OUTPUT_FILE);

    if (!outputFile.is_open()) {
        cerr << "Error opening file for writing!" << endl;
        return -1;
    }
    outputFile << "distance,order,cycle,greedy,edx" << endl;

    cityCount = distance_matrix.size();
    vector<citizen> originalPopulation;
    for (int i = 0; i < POPULATION_SIZE; i++) {
        originalPopulation.push_back(makeCitizen(createInitialPath(distance_matrix), distance_matrix));
    }

    auto tokenCount = 100 / TOKENVALUE;
    for (int i = 0; i <= tokenCount; i++) {
        for (int j = 0; j <= tokenCount; j++) {
            for (int z = 0; z <= tokenCount; z++) {
                auto u = tokenCount - i - j - z;
                if (u < 0) {
                    continue;
                }

                auto operatorSequence = makeOperatorSequence(i, j, z, u);
                auto operatorDistribution = makeOperatorSelection(i, j, z, u);
                for (int tries = 0; tries < TRIES; tries++) {
                    auto population = originalPopulation;
                    auto counter = 0;
                    while (counter++ != GENERATIONS) {
                        auto parentDistribution = makeRoulletteWheel(population);
                        auto newPopulation = makeNewPopulation(population, parentDistribution, operatorSequence, distance_matrix);
                        auto fullPopulation = combineLists(population, newPopulation);
                        auto mutatedPopulation = mutatePopulation(fullPopulation, distance_matrix);
                        sortByFitness(fullPopulation);
                        keepFittest(fullPopulation);
                        population = fullPopulation;
                    }

                    outputFile << population[0].second << "," << i * TOKENVALUE << "," << j * TOKENVALUE << "," << z * TOKENVALUE << "," << u * TOKENVALUE << endl;
                }
            }
        }
    }
    outputFile.close();

    cout << "Finished calculations!\n";
}
