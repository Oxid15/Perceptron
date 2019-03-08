#include "Layer.cpp"

enum metrics { accuracy, meanEuclidNorm };

enum taskType { bin_classification, regression };

template<typename T = float>
class NeuralNet
{
	bool empty;
	int layers;
	int matrixes;
	expArray<Layer<T>> arrLayers;
	expArray<Matrix<T>> arrMatrixes;
	functionType type;
	int out_len;
	T*net_out;
	T trainEff;
	T testEff;

	void backpropagation(T* target, T speed)
	{
		arrLayers[layers - 1]->setError(target, arrMatrixes[layers - 2], type);
		for (int i = layers - 2; i > 0; i--)
		{
			arrLayers[i]->setError(arrLayers[i + 1]->getError(), arrMatrixes[i], arrMatrixes[i - 1], type);
		}

		for (int i = 0; i < matrixes; i++)
		{
			arrMatrixes[i]->setWeights(arrLayers[i + 1]->getError(), arrLayers[i + 1]->getInput(), speed);
		}

		for (int i = 1; i < layers; i++)
		{
			for (int j = 0; j < arrLayers[i]->getNeuronsNum(); j++)
			{
				arrLayers[i]->getNeurons()[j]->addToBias(arrLayers[i]->getError(j), speed);
			}
		}
	}

	T* process(T* input)
	{
		arrLayers[0]->process(input);
		for (int i = 1; i < layers; i++)
		{
			arrLayers[i]->process(arrLayers[i - 1]->getOutput(), arrMatrixes[i - 1], type);
		}
		return arrLayers[layers - 1]->getOutput();
	}

public:

	NeuralNet()
	{
		layers = 2;
		matrixes = 1;
		type = sigmoid;
		empty = true;
		out_len = 1;
		net_out = new T;
	}

	NeuralNet(std::string fileName)
	{
		std::ifstream configFile(fileName);
		configFile >> layers;
		matrixes = layers - 1;

		int _type;
		configFile >> _type;
		type = functionType(_type);

		empty = false;

		int* neurons = new int[layers];
		for (int i = 0; i < layers; i++)
		{
			configFile >> neurons[i];
		}

		T** biases = new T*[layers];
		for (int i = 0; i < layers; i++)
		{
			biases[i] = new T[neurons[i]];
			for (int j = 0; j < neurons[i]; j++)
			{
				configFile >> biases[i][j];
			}
		}

		T** weights = new T*[matrixes];
		for (int i = 0; i < matrixes; i++)
		{
			weights[i] = new T[neurons[i] * neurons[i + 1]];
			for (int j = 0; j < neurons[i] * neurons[i + 1]; j++)
			{
				configFile >> weights[i][j];
			}
		}

		initialize(layers, matrixes, type, neurons, biases, weights);
	}

	~NeuralNet() { delete net_out; }

	void initialize(int _layers, int _matrixes, functionType _type, int* neurons, T** biases, T** weights)
	{
		layers = _layers;
		matrixes = _matrixes;
		type = _type;

		for (int i = 0; i < layers; i++)
		{
			if (i == 0)
			{
				Layer<T>* newLayer = new Layer<T>(biases[i], 1, neurons[i], neurons[i + 1]);
				arrLayers.add(*newLayer, i);
				continue;
			}

			if (i == layers - 1)
			{
				Layer<T>* newLayer = new Layer<T>(biases[i], neurons[i - 1], neurons[i], 1);
				arrLayers.add(*newLayer, i);
				continue;
			}
			Layer<T>* newLayer = new Layer<T>(biases[i], neurons[i - 1], neurons[i], neurons[i + 1]);
			arrLayers.add(*newLayer, i);
		}

		for (int i = 0; i < matrixes; i++)
		{
			arrMatrixes.add(*new Matrix<T>(weights[i], neurons[i + 1], neurons[i]), i);
		}

		out_len = arrLayers[layers - 1]->getNeuronsNum();
		net_out = new T[out_len];
	}

	T validate(
		std::string dataFileName,
		std::string resFileName,
		int size,
		metrics metric = metrics::accuracy,
		taskType type = taskType::bin_classification
	)
	{
		std::fstream data(dataFileName);
		std::fstream res(resFileName);

		T** output = new T*[size];
		for (int i = 0; i < size; i++)
			output[i] = new T[out_len];

		T** target_out = new T*[size];
		for (int i = 0; i < size; i++)
			target_out[i] = new T[out_len];

		for (int i = 0; i < size; i++)
		{
			int length = arrLayers[0]->getNeuronsNum();
			T* out = new T[out_len];
			out = process(readStrCsv<T>(data, length));

			for (int j = 0; j < out_len; j++)
			{
				output[i][j] = out[j];
			}

			target_out[i] = readStrCsv<T>(res, out_len);

			switch (type)
			{
			case(taskType::bin_classification):
			{
				if (output[i][0] >= 0.5)
					output[i][0] = 1;
				else
					output[i][0] = 0;
			}
			}
		}

		switch (metric)
		{
		case (metrics::accuracy):
		{
			int right = 0;
			int all = size;

			for (int i = 0; i < size; i++)
			{
				if (output[i][0] == target_out[i][0])
					right++;
			}
			T acc;
			acc = T(right) / T(all);
			return acc;
		}
		case (metrics::meanEuclidNorm):
		{
			T* distances = new T[size];
			for (int i = 0; i < size; i++)
			{
				distances[i] = euclidNorm<T>(target_out[i], output[i], out_len);
			}
			return mean<T>(distances, size);
		}
		}
		return NULL;
	}

	T** dataProcess(std::string fileName, int size)
	{
		std::fstream set(fileName);
		int out_len = arrLayers[layers - 1]->getNeuronsNum();
		T** net_out = new T*[out_len];
		for (int i = 0; i < out_len; i++)
			net_out[i] = new T;

		for (int i = 0; i < size; i++)
		{
			int length = arrLayers[0]->getNeuronsNum();
			net_out[i] = process(readStrCsv<T>(set, length));
		}
		return net_out;
	}

	void fit(
		std::string trainDataFName,
		std::string trainResFName,
		std::string testDataFName,
		std::string testResFName,
		int trainFileSize,
		int testFileSize,
		int epochs,
		T speed = 1,
		metrics metric = metrics::accuracy,
		taskType type = taskType::bin_classification,
		bool trainValidation = true,
		bool testValidation = false
	)
	{
		for (int k = 0; k < epochs; k++)
		{
			std::fstream data(trainDataFName);
			std::fstream res(trainResFName);
			for (int i = 0; i < trainFileSize; i++)
			{
				int length = arrLayers[0]->getNeuronsNum();
				net_out = process(readStrCsv<T>(data, length));

				T* target_out = new T[out_len];
				target_out = readStrCsv<T>(res, out_len);

				backpropagation(target_out, speed);

				if (trainValidation)
					trainEff = validate(trainDataFName, trainResFName, trainFileSize, metric, type);
				if (testValidation)
					testEff = validate(testDataFName, testResFName, testFileSize, metric, type);
			}
		}
	}

	void addLayer(int neurons, T maxWeight, T minWeight = 0, int seed = 0)
	{
		T* biases = new T[neurons];
		for (int i = 0; i < neurons; i++) { biases[i] = 0; }

		int prevNum = arrLayers[this->getLayersNum() - 2]->getNeuronsNum();
		int nextNum = arrLayers[this->getLayersNum() - 1]->getNeuronsNum();

		Layer<T>* layer = new Layer<T>(biases, prevNum, neurons, nextNum);
		arrLayers.add(*layer, layers - 1);
		layers++;

		arrMatrixes.del(matrixes - 1);
		matrixes--;

		Matrix<T>* matrix1 = new Matrix<T>(neurons, prevNum, seed, maxWeight, minWeight);
		Matrix<T>* matrix2 = new Matrix<T>(nextNum, neurons, seed, maxWeight, minWeight);
		arrMatrixes.add(*matrix1);
		arrMatrixes.add(*matrix2);
		matrixes += 2;
	}

	void addNeuron(int layer, T maxWeight, T minWeight = 0, int seed = 0)
	{
		if (layer > 0 and layer < layers)
		{
			int prevNum = arrLayers[layer]->getPrevNum();
			int nextNum = arrLayers[layer]->getNextNum();
			int neurons = arrLayers[layer]->getNeuronsNum();

			arrLayers[layer]->add();

			arrMatrixes.del(layer);
			arrMatrixes.del(layer - 1);

			Matrix<T>* matrix1 = new Matrix<T>(neurons + 1, prevNum, seed, maxWeight, minWeight);
			Matrix<T>* matrix2 = new Matrix<T>(nextNum, neurons + 1, seed, maxWeight, minWeight);

			arrMatrixes.add(*matrix1);
			arrMatrixes.add(*matrix2);
		}
	}

	void delLayer(int index, T maxWeight, T minWeight = 0, int seed = 0)
	{
		if (index > 0 and index < layers)
		{
			int prevNum = arrLayers[layers - 2]->getNeuronsNum();
			int nextNum = arrLayers[this->getLayersNum() - 1]->getNeuronsNum();

			arrLayers.del(index);
			layers--;

			arrMatrixes.del(matrixes - 1);
			arrMatrixes.del(matrixes - 2);
			matrixes -= 2;

			Matrix<T>* matrix = new Matrix<T>(nextNum, prevNum, seed, maxWeight, minWeight);
			arrMatrixes.add(*matrix);
			matrixes++;
		}
	}

	void delNeuron(int layer, T maxWeight, T minWeight = 0, int seed = 0)
	{
		if (layer > 0 and layer < layers)
		{
			int prevNum = arrLayers[layer]->getPrevNum();
			int nextNum = arrLayers[layer]->getNextNum();
			int neurons = arrLayers[layer]->getNeuronsNum();

			arrLayers[layer]->del(0);

			arrMatrixes.del(layer);
			arrMatrixes.del(layer - 1);

			Matrix<T>* matrix1 = new Matrix<T>(neurons - 1, prevNum, seed, maxWeight, minWeight);
			Matrix<T>* matrix2 = new Matrix<T>(nextNum, neurons - 1, seed, maxWeight, minWeight);

			arrMatrixes.add(*matrix1);
			arrMatrixes.add(*matrix2);
		}
	}

	expArray<Matrix<T>> getMatrixes() { return arrMatrixes; }

	Layer<T>* getLayers() { return arrLayers.getArr(); }

	int getMatrixesNum() { return matrixes; }

	int getLayersNum() { return layers; }

	T getEff() { return trainEff; }

	T* getOutput() { return net_out; }

	void fileOutput(std::string fileName)
	{
		std::ofstream file(fileName);
		file << layers << "\n";

		int i = int(type);
		file << i << "\n";

		for (int i = 0; i < layers; i++)
		{
			file << arrLayers[i]->getNeuronsNum();
			if (i != layers - 1)
				file << " ";
		}
		file << "\n";

		for (int i = 0; i < layers; i++)
		{
			for (int j = 0; j < arrLayers[i]->getNeuronsNum(); j++)
			{
				Neuron<T>* tmp = arrLayers[i]->getNeurons()[j];
				file << std::to_string(tmp->getBias()) << " ";
			}
			file << "\n";
		}

		for (int k = 0; k < matrixes; k++)
		{
			int height = arrMatrixes[k]->getHeight();
			int lenght = arrMatrixes[k]->getLength();

			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < lenght; j++)
				{
					file << std::to_string(arrMatrixes[k]->getWeight(j, i)) << " ";
				}
			}
			file << "\n";
		}
	}

	void weightsOutput(std::string fileName)
	{
		std::ofstream file(fileName, std::ios::app);

		for (int i = 1; i < layers; i++)
		{
			for (int j = 0; j < arrLayers[i]->getNeuronsNum(); j++)
			{
				file << std::to_string(arrLayers[i]->getNeurons()[j]->getBias()) << ";";
			}
		}

		for (int i = 0; i < matrixes; i++)
		{
			arrMatrixes[i]->fileOutput(file);
		}
		file << "\n";
	}
};