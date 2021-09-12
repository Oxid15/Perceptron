// Templates with type T is used to allow using different types of numbers, not only double
// double by default in NeuralNet class

#include "Layer.cpp"

enum metrics { accuracy, meanEuclidNorm };

enum taskType { bin_classification, regression };

template<typename T = double>
class NeuralNet
{
	bool isEmpty;
	int layersNum;
	int matrixesNum;
	expArray<Layer<T>> arrLayers;
	expArray<Matrix<T>> arrMatrixes;
	functionType ftype;
	int outputLen;
	T* net_out;
	T trainEff;
	T testEff;

	void backpropagation(T* targetOutput, T speed)
	{
		arrLayers[layersNum - 1]->setError(targetOutput, arrMatrixes[layersNum - 2], ftype);
		for (int i = layersNum - 2; i > 0; i--)
		{
			arrLayers[i]->setError(arrLayers[i + 1]->getError(), arrMatrixes[i], arrMatrixes[i - 1], ftype);
		}

		for (int i = 0; i < matrixesNum; i++)
		{
			arrMatrixes[i]->setWeights(arrLayers[i + 1]->getError(), arrLayers[i + 1]->getInput(), speed);
		}

		for (int i = 1; i < layersNum; i++)
		{
			for (int j = 0; j < arrLayers[i]->getNeuronsNum(); j++)
			{
				arrLayers[i]->getNeurons()[j]->addToBias(arrLayers[i]->getError(j), speed);
			}
		}
	}

public:

	T* process(T* input)
	{
		arrLayers[0]->process(input);
		for (int i = 1; i < layersNum; i++)
		{
			arrLayers[i]->process(arrLayers[i - 1]->getOutput(), arrMatrixes[i - 1], ftype);
		}
		return arrLayers[layersNum - 1]->getOutput();
	}

	NeuralNet(std::string fileName)
	{
		std::ifstream configFile(fileName);
		configFile >> layersNum;
		matrixesNum = layersNum - 1;

		int _ftype;
		configFile >> _ftype;
		ftype = functionType(_ftype);

		isEmpty = false;

		int* neurons = new int[layersNum];
		for (int i = 0; i < layersNum; i++)
		{
			configFile >> neurons[i];
		}

		T** biases = new T*[layersNum];
		for (int i = 0; i < layersNum; i++)
		{
			biases[i] = new T[neurons[i]];
			for (int j = 0; j < neurons[i]; j++)
			{
				configFile >> biases[i][j];
			}
		}

		T** weights = new T*[matrixesNum];
		for (int i = 0; i < matrixesNum; i++)
		{
			weights[i] = new T[neurons[i] * neurons[i + 1]];
			for (int j = 0; j < neurons[i] * neurons[i + 1]; j++)
			{
				configFile >> weights[i][j];
			}
		}

		initialize(layersNum, matrixesNum, ftype, neurons, biases, weights);
	}

	~NeuralNet() { delete net_out; }

	void initialize(int _layersNum, int _matrixesNum, functionType _ftype, int* neurons, T** biases, T** weights)
	{
		layersNum = _layersNum;
		matrixesNum = _matrixesNum;
		ftype = _ftype;

		for (int i = 0; i < layersNum; i++)
		{
			if (i == 0)
			{
				Layer<T>* newLayer = new Layer<T>(biases[i], 1, neurons[i], neurons[i + 1]);
				arrLayers.add(*newLayer, i);
				continue;
			}

			if (i == layersNum - 1)
			{
				Layer<T>* newLayer = new Layer<T>(biases[i], neurons[i - 1], neurons[i], 1);
				arrLayers.add(*newLayer, i);
				continue;
			}
			Layer<T>* newLayer = new Layer<T>(biases[i], neurons[i - 1], neurons[i], neurons[i + 1]);
			arrLayers.add(*newLayer, i);
		}

		for (int i = 0; i < matrixesNum; i++)
		{
			arrMatrixes.add(*new Matrix<T>(weights[i], neurons[i + 1], neurons[i]), i);
		}

		outputLen = arrLayers[layersNum - 1]->getNeuronsNum();
		net_out = new T[outputLen];
	}

	T validate(
		std::string dataFileName,
		std::string resFileName,
		int fileSize,                                //it should be equal for both files
		metrics metric = metrics::accuracy,
		taskType type = taskType::bin_classification
	)
	{
		std::fstream data(dataFileName);
		std::fstream res(resFileName);

		T** output = new T*[fileSize];
		for (int i = 0; i < fileSize; i++)
			output[i] = new T[outputLen];

		T** target_out = new T*[fileSize];
		for (int i = 0; i < fileSize; i++)
			target_out[i] = new T[outputLen];

		for (int i = 0; i < fileSize; i++)
		{
			int length = arrLayers[0]->getNeuronsNum();
			T* out = new T[outputLen];
			T* in = new T[length];
			readStrCsv(data, in, length);
			out = process(in);

			for (int j = 0 ;j < outputLen; j++)
			{
				output[i][j] = out[j];
			}

			readStrCsv<T>(res, target_out[i], outputLen);

			switch (type)
			{
			case(taskType::bin_classification):			
			{
				//TODO: review this block
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
			for (int i = 0; i < fileSize; i++)
			{
				if (output[i][0] == target_out[i][0])
					right++;
			}

			int all = fileSize;
			return T(right) / T(all);
		}
		case (metrics::meanEuclidNorm):
		{
			T* distances = new T[fileSize];
			for (int i = 0; i < fileSize; i++)
			{
				distances[i] = euclidDist<T>(target_out[i], output[i], outputLen);
			}
			return mean<T>(distances, fileSize);
		}
		}
		return NULL;
	}

	T** dataProcess(std::string fileName, int size)
	{
		std::fstream set(fileName);
		int outputLen = arrLayers[layersNum - 1]->getNeuronsNum();
		T** net_out = new T*[outputLen];
		for (int i = 0; i < outputLen; i++)
			net_out[i] = new T;

		for (int i = 0; i < size; i++)
		{
			int length = arrLayers[0]->getNeuronsNum();
			T* input = new T[length];
			readStrCsv(set, input, length);
			net_out[i] = process(input);
		}
		return net_out;
	}

	void fit(
		std::string trainDataFName,
		int trainDataFSize,
		std::string trainResFName,
		int trainResFSize,
		std::string testDataFName,
		int testDataFSize,
		std::string testResFName,
		int testResFSize,
		int epochs,
		T speed = 1,
		metrics metric = metrics::accuracy,
		taskType ftype = taskType::bin_classification,
		bool trainValidation = true,
		bool testValidation = false
	)
	{
		for (int k = 0; k < epochs; k++)
		{
			std::fstream data(trainDataFName);
			std::fstream res(trainResFName);
			for (int i = 0; i < trainDataFSize; i++)
			{
				int length = arrLayers[0]->getNeuronsNum();
				T* input = new T[length];
				readStrCsv<T>(data, input, length);
				net_out = process(input);

				T* target_out = new T[outputLen];
				readStrCsv<T>(res, target_out, outputLen);

				backpropagation(target_out, speed);

				if (trainValidation)
					trainEff = validate(trainDataFName, trainResFName, trainDataFSize, metric, ftype);  //FIXME
				if (testValidation)
					testEff = validate(testDataFName, testResFName, testDataFSize, metric, ftype);

				delete target_out;
				delete input;
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
		arrLayers.add(*layer, layersNum - 1);
		layersNum++;

		arrMatrixes.del(matrixesNum - 1);
		matrixesNum--;

		Matrix<T>* matrix1 = new Matrix<T>(neurons, prevNum, seed, maxWeight, minWeight);
		Matrix<T>* matrix2 = new Matrix<T>(nextNum, neurons, seed, maxWeight, minWeight);
		arrMatrixes.add(*matrix1);
		arrMatrixes.add(*matrix2);
		matrixesNum += 2;
	}

	void addNeuron(int layer, T maxWeight, T minWeight = 0, int seed = 0)
	{
		if (layer > 0 and layer < layersNum)
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
		if (index > 0 and index < layersNum)
		{
			int prevNum = arrLayers[layersNum - 2]->getNeuronsNum();
			int nextNum = arrLayers[this->getLayersNum() - 1]->getNeuronsNum();

			arrLayers.del(index);
			layersNum--;

			arrMatrixes.del(matrixesNum - 1);
			arrMatrixes.del(matrixesNum - 2);
			matrixesNum -= 2;

			Matrix<T>* matrix = new Matrix<T>(nextNum, prevNum, seed, maxWeight, minWeight);
			arrMatrixes.add(*matrix);
			matrixesNum++;
		}
	}

	void delNeuron(int layer, T maxWeight, T minWeight = 0, int seed = 0)
	{
		if (layer > 0 and layer < layersNum)
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

	expArray <Layer<T>> getLayers() { return arrLayers; }

	int getMatrixesNum() { return matrixesNum; }

	int getLayersNum() { return layersNum; }

	T getEff() { return trainEff; }

	T* getOutput() { return net_out; }

	void fileOutput(std::string fileName)
	{
		std::ofstream file(fileName);
		file << layersNum << "\n";

		int i = int(ftype);
		file << i << "\n";

		for (int i = 0; i < layersNum; i++)
		{
			file << arrLayers[i]->getNeuronsNum();
			if (i != layersNum - 1)
				file << " ";
		}
		file << "\n";

		for (int i = 0; i < layersNum; i++)
		{
			for (int j = 0; j < arrLayers[i]->getNeuronsNum(); j++)
			{
				Neuron<T>* tmp = arrLayers[i]->getNeurons()[j];
				file << std::to_string(tmp->getBias()) << " ";
			}
			file << "\n";
		}

		for (int k = 0; k < matrixesNum; k++)
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

		for (int i = 1; i < layersNum; i++)
		{
			for (int j = 0; j < arrLayers[i]->getNeuronsNum(); j++)
			{
				file << std::to_string(arrLayers[i]->getNeurons()[j]->getBias()) << ";";
			}
		}

		for (int i = 0; i < matrixesNum; i++)
		{
			arrMatrixes[i]->fileOutput(file);
		}
		file << "\n";
	}
};