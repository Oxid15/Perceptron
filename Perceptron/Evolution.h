#include"Perceptron.cpp"

template<typename T>
void setrandomWeights(NeuralNet<T>& net, std::default_random_engine engine, int seed, T maxWeight, T minWeight = 0)
{
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

template<typename T>
class Population
{
	NeuralNet<T>* population;
	int size;
	int inputSize;			 
	int outputSize;			 
	std::default_random_engine engine;

public:

	Population()
	{
		size = 1;
		population = new NeuralNet<T>;
	}

	Population(int _inputSize, int _outputSize, T maxWeight, T minWeight = 0, int maxLayers = 3, int minLayers = 2,functionType type = sigmoid, int seed = 0, int population_size = 10)
	{
		size = population_size;
		inputSize = _inputSize;
		outputSize = _outputSize;

		population = new NeuralNet<T>[size];

		static std::default_random_engine engine;

		int layers = randomNumber<T>(seed, engine, maxLayers + 1, minLayers);
		int matrixes = layers - 1;
		int* neurons = new int[layers];

		neurons[0] = inputSize;
		neurons[layers - 1] = outputSize;
		for (int i = 1; i < layers - 1; i++)
		{
			neurons[i] = inputSize + int(randomNumber<T>(seed, engine, log(inputSize) + 2));
		}

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
			population[i].initialize(2, 1, type, neurons, biases, weights);
		}
	}

	void mutation(float mutation_chance = 0.05, T maxWeight = 1, T minWeight = 0, int seed = 0)
	{
		for (int i = 0; i < size; i++)
		{
			float chance = randomNumber<T>(seed, engine, 1);

			if (chance <= mutation_chance)
			{
				float mut_prop = randomNumber<T>(seed, engine, 1);

				if (population[i].getLayersNum() < 3)
				{
					int prevLayerNeurons = population[i].getLayers()[0].getNeuronsNum();
					population[i].addLayer(prevLayerNeurons, maxWeight, minWeight, seed);
				}
				else
				{
					if(mut_prop <= 0.16)
					{
						int prevLayerNeurons = population[i].getLayers()[0].getNeuronsNum();
						population[i].addLayer(prevLayerNeurons, maxWeight, minWeight, seed);
					}
					else if(mut_prop > 0.16 and mut_prop <= 0.33)
					{
						int maxLayers = population[i].getLayersNum();
						int layer = randomNumber<T>(seed, engine, maxLayers - 1, 1);
						population[i].delLayer(layer, maxWeight, minWeight, seed);
					}
					else if(mut_prop > 0.33 and mut_prop <= 0.66)
					{
						int maxLayers = population[i].getLayersNum();
						int layer = randomNumber<T>(seed, engine, maxLayers - 1, 1);
						population[i].addNeuron(layer, maxWeight, minWeight, seed);
					}
					else if(mut_prop > 0.66)
					{
						int maxLayers = population[i].getLayersNum();
						int layer = randomNumber<T>(seed, engine, maxLayers - 1, 1);
						population[i].delNeuron(layer, maxWeight, minWeight, seed);
					}
				}
			}
		}
	}

	void fileOutput(std::string fileName)
	{
		//.........
	}

};