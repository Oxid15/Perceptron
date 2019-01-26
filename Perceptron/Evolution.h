#include"Perceptron.cpp"

template<typename T>
void setrandomWeights(NeuralNet<T>& net, int seed, T maxWeight, T minWeight = 0)
{
	static std::default_random_engine engine;
	int layers = net.getLayersNum();
	for (int i = 0; i < layers; i++)
	{
		int neurons = net.getLayers()[i]->getNeuronsNum();
		for (int j = 0; j < neurons; j++)
		{
			if (i)
				net.getLayers()[i]->getNeurons()[j]->setBias(randomNumber<T>(seed, engine, maxWeight, minWeight));
			else
				net.getLayers()[i]->getNeurons()[j]->setBias(0);
			seed++;
		}
	}

	int matrixes = net.getMatrixesNum();
	for (int k = 0; k < matrixes; k++)
	{
		int length = net.getMatrixes()[k]->getLength();
		int height = net.getMatrixes()[k]->getHeight();
		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < height; j++)
			{
				net.getMatrixes()[k]->setWeight(i, j, randomNumber<T>(seed, engine, maxWeight, minWeight));
				seed++;
			}
		}
	}
}

template <typename T>
NeuralNet<T>* init(int inputSize, int outputSize, T maxWeight, T minWeight = 0, int seed = 42, int population_size = 10)
{
	static std::default_random_engine engine;
	NeuralNet<T>* population = new NeuralNet<T>[population_size];

	int layers = population[0].getLayersNum();
	int matrixes = population[0].getMatrixesNum();
	int neurons[2] = { inputSize, outputSize };

	T** biases = new T*[layers];
	for (int i = 0; i < layers; i++)
	{
		biases[i] = new T[neurons[i]];
		for (int j = 0; j < neurons[i]; j++)
		{
			biases[i][j] = randomNumber<T>(seed, engine, maxWeight, minWeight);
			seed++;
		}
	}

	T** weights = new T*[matrixes];
	for (int i = 0; i < matrixes; i++)
	{
		weights[i] = new T[neurons[i] * neurons[i + 1]];
		for (int j = 0; j < neurons[i] * neurons[i + 1]; j++)
		{
			weights[i][j] = randomNumber<T>(seed, engine, maxWeight, minWeight);
			seed++;
		}
	}

	for (int i = 0; i < population_size; i++)
	{
		population[i].initialize(2, 1, functionType::sigmoid, neurons, biases, weights);
	}
	return population;
}

template <typename T>
void mutation(NeuralNet<T>* population, int population_size = 10, float mutation_chance = 0.05, int seed = 42)
{
	static std::default_random_engine engine;
	for (int i = 0; i < population_size; i++)
	{
		float chance = randomNumber<T>(seed, engine, 1);
		if (chance <= mutation_chance)
		{
			float mut_prop = randomNumber<T>(seed, engine, 1);
			if (mut_prop >= 0.5)
			{
				//adds layer	
			}
			else
			{
				//adds neurons to random layer
			}
		}
	}
}