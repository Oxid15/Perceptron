#include"computations.h"
#include"files.h"

enum metrics { accuracy, meanEuclidNorm };

enum taskType { bin_classification, regression };

template<typename T>
class Neuron
{
	T bias;
	T* in;
	T out;
	int prevNum;
	int nextNum;

	template<typename T>
	T activation(T num, functionType type)
	{
		switch (type)
		{
		case sigmoid:
			return sig(num);
		case softpls:
			return softplus(num);
		}
	}

public:

	Neuron(int _prevNum = 1, int _nextNum = 1, T _bias = 0)
	{
		bias = _bias;
		prevNum = _prevNum;
		nextNum = _nextNum;
		in = new T[prevNum];
	}

	~Neuron() {	delete in; }

	T process(T* _in, T* weights, functionType type)
	{
		in = _in;
		out = activation(weighedSum(in, weights, prevNum) + bias, type);
		return out;
	}

	T process(T in)
	{
		out = in;
		return out;
	}

	void addToBias(T error, T speed) { bias += speed * error; }

	void setBias(T _bias) { bias = _bias; }

	T getBias() { return bias; }

	T getOutput() { return out; }

	int getNextNum() { return nextNum; }

	int getPrevNum() { return prevNum; }
};

template<typename T>
class AdjMatrix
{
	T** arr;
	int length;
	int height;

public:
	AdjMatrix()
	{
		length = 1;
		height = 1;
		arr = new T*[length];
		for (int i = 0; i < length; i++)
		{
			arr[i] = new T[height];
		}
	}

	AdjMatrix(int _length, int _height, int seed, T maxWeight, T minWeight)
	{
		length = _length;
		height = _height;
		arr = new T*[length];
		for (int i = 0; i < length; i++)
		{
			arr[i] = new T[height];
		}

		int k = 0;
		std::default_random_engine engine;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < length; j++)
			{
				arr[j][i] = randomNumber<T>(seed, engine, maxWeight, minWeight);
				k++;
			}
		}
	}

	AdjMatrix(T* weights, int _length = 1, int _height = 1)
	{
		length = _length;
		height = _height;
		arr = new T*[length];
		for (int i = 0; i < length; i++)
		{
			arr[i] = new T[height];
		}

		int k = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < length; j++)
			{
				arr[j][i] = weights[k];
				k++;
			}
		}
	}

	~AdjMatrix() 
	{
		for (int i = 0; i < length; i++)
		{
			delete arr[i];
		}
		delete arr;
	}

	void fileOutput(std::ofstream& file)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < length; j++)
			{
				file << arr[j][i] << ";";
			}
		}
	}

	T* getStrWeights(int index)
	{
		T* temp = new T[length];
		for (int i = 0; i < length; i++)
		{
			temp[i] = arr[i][index];
		}
		return temp;
	}

	T* getColWeights(int index)
	{
		T* temp = new T[height];
		for (int i = 0; i < height; i++)
		{
			temp[i] = arr[index][i];
		}
		return temp;
	}

	void setWeight(int i, int j, T weight) { arr[i][j] = weight; }

	void setWeights(T* layerError, T* layerInput, T speed)
	{
		T** dWeights = new T*[length];
		for (int i = 0; i < length; i++)
		{
			dWeights[i] = new T[height];
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < length; j++)
			{
				dWeights[j][i] = (speed * layerInput[i] * layerError[j]);
			}
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < length; j++)
			{
				arr[j][i] += dWeights[j][i];
			}
		}
	}

	T getWeight(int i, int j) { return arr[i][j]; }
	int getLength() { return length; }
	int getHeight() { return height; }
};

template<typename T>
class Layer
{
	expArray<Neuron<T>> arr;
	T* input;
	T* output;
	T* error;
	int neurons;
	int prevNum;
	int nextNum;
public:
	Layer()
	{
		input = new T;
		output = new T;
		error = new T;
		neurons = 1;
		prevNum = 1;
		nextNum = 1;
	}

	Layer(T* biases, int _prevNum = 1, int _neuronsNum = 1, int _nextNum = 1)
	{
		neurons = _neuronsNum;
		prevNum = _prevNum;
		nextNum = _nextNum;
		input = new T[prevNum];
		output = new T[neurons];
		error = new T[neurons];

		for (int i = 0; i < neurons; i++)
			arr.add(*new Neuron<T>(prevNum, _nextNum, biases[i]));
	}

	~Layer()
	{
		delete input;
		delete output;
		delete error;
	}

	T* process(T* _input)
	{
		input = _input;
		for (int i = 0; i < neurons; i++)
		{
			output[i] = arr[i]->process(_input[i]);
		}
		return output;
	}

	T* process(T* _input, AdjMatrix<T>* matrix, functionType type)
	{
		input = _input;
		for (int i = 0; i < neurons; i++)
		{
			output[i] = arr[i]->process(_input, matrix->getColWeights(i), type);
		}
		return output;
	}

	void setError(T* target, AdjMatrix<T>* matrix, functionType funcType)
	{
		for (int i = 0; i < neurons; i++)
		{
			error[i] = (target[i] - output[i]) *
				derivative<T>(funcType, weighedSum<T>(input, matrix->getColWeights(i), prevNum) + arr[i]->getBias());
		}

	}

	void setError(T* errors, AdjMatrix<T>* thisMatrix, AdjMatrix<T>* prevMatrix, functionType funcType)
	{
		for (int i = 0; i < neurons; i++)
		{
			error[i] = (weighedSum(errors, thisMatrix->getStrWeights(i), arr[i]->getNextNum())) *
				(derivative<T>(funcType, weighedSum<T>(input, prevMatrix->getColWeights(i), prevNum) + arr[i]->getBias()));
		}
	}

	void setNeurons(int _neurons) { neurons = _neurons; }

	void add()
	{
		Neuron<T>* neuron = new Neuron<T>(prevNum, nextNum, 0);
		arr.add(*neuron);
		neurons++;
	}

	void del(int index)
	{
		arr.del(index);
		neurons--;
	}

	T* getInput() { return input; }

	T* getOutput() { return output; }

	T* getError() { return error; }

	T getError(int index) { return error[index]; }

	expArray<Neuron<T>> getNeurons() { return arr; }

	int getNextNum() { return nextNum; }

	int getPrevNum() { return prevNum; }

	int getNeuronsNum() { return neurons; }
};

template<typename T>
class NeuralNet
{
	bool empty;
	int layers;
	int matrixes;
	expArray<Layer<T>> arrLayers;
	expArray<AdjMatrix<T>> arrMatrixes;
	functionType type;
	int out_len;
	T*net_out;
	T valEff;

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
		switch (_type)
		{
		case 0:
			type = sigmoid;
			break;
		case 1:
			type = softpls;
		}

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

	~NeuralNet() {	delete net_out;	}

	void initialize(int _layers, int _matrixes, functionType _type, int* neurons, T** biases, T** weights)
	{
		layers = _layers;
		matrixes = _matrixes;
		type = _type;

		for (int i = 0; i < layers; i++)
		{
			if (i == 0)
			{
				arrLayers.add(*new Layer<T>(biases[i], 1, neurons[i], neurons[i + 1]), i);
				continue;
			}

			if (i == layers - 1)
			{
				arrLayers.add(*new Layer<T>(biases[i], neurons[i - 1], neurons[i], 1), i);
				continue;
			}
			arrLayers.add(*new Layer<T>(biases[i], neurons[i - 1], neurons[i], neurons[i + 1]), i);
		}

		for (int i = 0; i < matrixes; i++)
		{
			arrMatrixes.add(*new AdjMatrix<T>(weights[i], neurons[i + 1], neurons[i]), i);
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
			acc = double(right) / double(all);
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
				std::string dataFileName, 
				std::string resFileName, 
				int size,
				int epochs, 
				T speed = 1, 
				metrics metric = metrics::accuracy, 
				taskType type = taskType::bin_classification
			)
	{
		for (int k = 0; k < epochs; k++)
		{
			std::fstream data(dataFileName);
			std::fstream res(resFileName);
			for (int i = 0; i < size; i++)
			{
				int length = arrLayers[0]->getNeuronsNum();
				net_out = process(readStrCsv<T>(data, length));

				T* target_out = new T[out_len];
				target_out = readStrCsv<T>(res, out_len);

				backpropagation(target_out, speed);

				valEff = validate(dataFileName, resFileName, size, metric, type);
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

		AdjMatrix<T>* matrix1 = new AdjMatrix<T>(neurons, prevNum, seed, maxWeight, minWeight);
		AdjMatrix<T>* matrix2 = new AdjMatrix<T>(nextNum, neurons, seed, maxWeight, minWeight);
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

			AdjMatrix<T>* matrix1 = new AdjMatrix<T>(neurons + 1, prevNum, seed, maxWeight, minWeight);
			AdjMatrix<T>* matrix2 = new AdjMatrix<T>(nextNum, neurons + 1, seed, maxWeight, minWeight);

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

			AdjMatrix<T>* matrix = new AdjMatrix<T>(nextNum, prevNum, seed, maxWeight, minWeight);
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

			AdjMatrix<T>* matrix1 = new AdjMatrix<T>(neurons - 1, prevNum, seed, maxWeight, minWeight);
			AdjMatrix<T>* matrix2 = new AdjMatrix<T>(nextNum, neurons - 1, seed, maxWeight, minWeight);

			arrMatrixes.add(*matrix1);
			arrMatrixes.add(*matrix2);
		}
	}

	expArray<AdjMatrix<T>> getMatrixes() { return arrMatrixes; }

	Layer<T>* getLayers() { return arrLayers.getArr(); }

	int getMatrixesNum() { return matrixes; }

	int getLayersNum() { return layers; }

	T getEff() { return valEff; }

	T* getOutput() { return net_out; }

	void fileOutput(std::string fileName)
	{
		std::ofstream file(fileName);
		file << layers << "\n";

		switch (type)
		{
		case sigmoid:
			file << "0\n";
			break;
		case softpls:
			file << "3\n";
		}

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
				Neuron<T> tmp = *arrLayers[i]->getNeurons()[j];
				file << std::to_string(tmp.getBias()) << " ";
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