#include"NeuralNet.cpp"

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
				net.getLayers()[i]->getNeurons()[j]->setBias(unifRealRandNum<T>(seed, engine, maxWeight, minWeight));
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
				net.getMatrixes()[k]->setWeight(i, j, unifRealRandNum<T>(seed, engine, maxWeight, minWeight));
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

	T maxWeight;
	T minWeight;

	T* results;

public:

	Population
	(
		int _inputSize, 
		int _outputSize, 
		T _maxWeight, 
		T _minWeight = 0, 
		int population_size = 10, 
		int maxLayers = 3, 
		functionType type = sigmoid, 
		int seed = 0
	)
	{
		size = population_size;
		inputSize = _inputSize;
		outputSize = _outputSize;
		maxWeight = _maxWeight;
		minWeight = _minWeight;

		results = new T[size];
		population = new NeuralNet<T>[size];

		static std::default_random_engine engine;

		int minLayers = 2;

		for (int i = 0; i < population_size; i++)
		{

			int layers = unifRealRandNum<T>(seed, engine, maxLayers + 1, minLayers);
			int matrixes = layers - 1;
			int* neurons = new int[layers];

			neurons[0] = inputSize;
			for (int i = 1; i < layers - 1; i++)
			{
				neurons[i] = inputSize + int(unifRealRandNum<T>(seed, engine, log(inputSize) + 2));
			}
			neurons[layers - 1] = outputSize;

			T** biases = new T*[layers];
			for (int i = 0; i < layers; i++)
			{
				biases[i] = new T[neurons[i]];
				for (int j = 0; j < neurons[i]; j++)
				{
					biases[i][j] = unifRealRandNum<T>(seed, engine, maxWeight, minWeight);
					seed++;
				}
			}

			T** weights = new T*[matrixes];
			for (int i = 0; i < matrixes; i++)
			{
				weights[i] = new T[neurons[i] * neurons[i + 1]];
				for (int j = 0; j < neurons[i] * neurons[i + 1]; j++)
				{
					weights[i][j] = unifRealRandNum<T>(seed, engine, maxWeight, minWeight);
					seed++;
				}
			}

			population[i].initialize(layers, matrixes, type, neurons, biases, weights);
		}
	}

	void mutation(float mutation_chance = 0.05, int seed = 0)
	{
		for (int i = 0; i < size; i++)
		{
			float chance = unifRealRandNum<T>(seed, engine, 1);

			if (chance <= mutation_chance)
			{
				float mut_prop = unifRealRandNum<T>(seed, engine, 1);

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
						int layer = unifRealRandNum<T>(seed, engine, maxLayers - 1, 1);
						if (population[i].getLayersNum() > 1)
							population[i].delLayer(layer, maxWeight, minWeight, seed);
					}
					else if(mut_prop > 0.33 and mut_prop <= 0.66)
					{
						int maxLayers = population[i].getLayersNum();
						int layer = unifRealRandNum<T>(seed, engine, maxLayers - 1, 1);
						population[i].addNeuron(layer, maxWeight, minWeight, seed);
					}
					else if(mut_prop > 0.66)
					{
						int maxLayers = population[i].getLayersNum();
						int layer = unifRealRandNum<T>(seed, engine, maxLayers - 1, 1);
						if (population[i].getLayers()[layer].getNeuronsNum() > 1)
							population[i].delNeuron(layer, maxWeight, minWeight, seed);
					}
				}
			}
		}
	}

	void evaluate(
				std::string trainDataFName,
				std::string trainResFName,
				std::string testDataFName,
				std::string testResFName,
				int trainFSize,
				int testFSize,
				int epochs,
				T speed = 1,
				metrics metric = metrics::accuracy,
				taskType type = taskType::bin_classification,
				bool trainValidation = true,
				bool testValidation = false
				)
	{
		for (int i = 0; i < size; i++)
		{
			population[i].fit
			(
				trainDataFName, 
				trainResFName, 
				testDataFName, 
				testResFName, 
				testFSize, 
				trainFSize, 
				epochs, 
				speed, 
				metric, 
				type, 
				trainValidation, 
				testValidation
			);
			results[i] = population[i].getEff();
		}
	}

	//void select()
	//{
	//	
	//}

	void fileOutput(FileName fileName)
	{
		for (int i = 0; i < size; i++)
		{
			std::string name = fileName.getName() + "_" + std::to_string(i) + '.' + fileName.getExt();
			population[i].fileOutput(name);
		}
	}

	T* getResults() { return results; }
};