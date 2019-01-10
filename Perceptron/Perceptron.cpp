#include<math.h>
#include<fstream>									  
#include<cstdlib>
#include<iostream>
#include<string>
#include<time.h>
#include"computing.h"

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

public:
	AdjMatrix()
	{
		length = 1;
		height = 1;
		arr = new T*;
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
		delete[] arr;
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

	void setWeight(int i, int j, T weight)
	{
		arr[i][j] = weight;
	}

	T getWeight(int i, int j)
	{
		return arr[i][j];
	}

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
	int lenInput;
public:
	Layer()
	{
		input = new T;
		output = new T;
		error = new T;
		neurons = 1;
		lenInput = 1;
	}

	Layer(T* biases, int prevNum = 1, int _neuronsNum = 1, int nextNum = 1)
	{
		neurons = _neuronsNum;
		lenInput = prevNum;
		input = new T[lenInput];
		output = new T[neurons];
		error = new T[neurons];

		for (int i = 0; i < neurons; i++)
			arr.add(*new Neuron<T>(prevNum, nextNum, biases[i]), i);
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
				derivative<T>(funcType, weighedSum<T>(input, matrix->getColWeights(i), lenInput) + arr[i]->getBias());
		}

	}

	void setError(T* errors, AdjMatrix<T>* thisMatrix, AdjMatrix<T>* prevMatrix, functionType funcType)
	{
		for (int i = 0; i < neurons; i++)
		{
			error[i] = (weighedSum(errors, thisMatrix->getStrWeights(i), arr[i]->getNextNum())) *
				(derivative<T>(funcType, weighedSum<T>(input, prevMatrix->getColWeights(i), lenInput) + arr[i]->getBias()));
		}
	}

	void setNeurons(int _neurons) { neurons = _neurons; }

	T* getInput() { return input; }

	T* getOutput() { return output; }

	T* getError() { return error; }

	T getError(int index) { return error[index]; }

	Neuron<T>* getNeurons()
	{
		return arr.getArr();
	}

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

	T* getStrFromFile(std::ifstream& trainSet)
	{
		int size = arrLayers[0]->getNeuronsNum();
		T* output = new T[size];
		for (int i = 0; i < size; i++)
		{
			trainSet >> output[i];
			trainSet.get();
		}
		return output;
	}

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

	void writeLog(T* output, T* target, std::ofstream& file)
	{
		int len = arrLayers[layers - 1]->getNeuronsNum();
		if (len != 1)
		{
			file << "\"";
			for (int i = 0; i < len; i++)
			{
				file << output[i] << ",";
			}
			file << "\";";

			file << "\"";
			for (int i = 0; i < len; i++)
			{
				file << target[i] << ",";
			}
			file << "\";\n";
		}
		else
		{
			file << output[0] << ";";
			file << target[0] << ";\n";
		}
	}

public:

	NeuralNet()
	{
		layers = 2;
		matrixes = 1;
		type = sigmoid;
		empty = true;
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
	}

	T* dataProcess(std::string fileName, int size)
	{
		std::ifstream set(fileName);
		std::ofstream log("testLog.csv");
		for (int i = 0; i < size; i++)
		{
			int out_len = arrLayers[layers - 1]->getNeuronsNum();
			T* net_out = new T[out_len];
			net_out = process(getStrFromFile(set));
		}
	}

	void train(std::string fileName, int size, int epochs, T speed)
	{
		for (int k = 0; k < epochs; k++)
		{
			std::ifstream trainSet(fileName);
			for (int i = 0; i < size; i++)
			{
				int out_len = arrLayers[layers - 1]->getNeuronsNum();
				T* net_out = new T[out_len];
				net_out = process(getStrFromFile(trainSet));

				T* target_out = new T[out_len];
				for (int i = 0; i < out_len; i++)
				{
					trainSet >> target_out[i];
					trainSet.get();
				}
				backpropagation(target_out, speed);
			}
		}
	}

	void addLayer(int neurons, T* biases)
	{
		layers += 1;
	}

	AdjMatrix<T>** getMatrixes() { return arrMatrixes; }

	Layer<T>** getLayers() { return arrLayers; }

	int getMatrixesNum() { return matrixes; }

	int getLayersNum() { return layers; }

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
				Neuron<T> tmp = arrLayers[i]->getNeurons()[j];
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

template<typename T>
T getAccuracyFromFile(std::string fileName, int output_length, int size)
{
	std::ifstream file(fileName);
	T all = 0;
	T correct = 0;
	T eff;
	for (int p = 0; p < size; p++)
	{
		for (int k = 0; k < output_length; k++)
		{
			T* output = new T[output_length];
			T* target = new T[output_length];
			for (int i = 0; i < output_length; i++)
			{
				file >> output[i];
				file.get();
			}
			for (int i = 0; i < output_length; i++)
			{
				file >> target[i];
				file.get();
			}
			for (int i = 0; i < output_length; i++)
			{
				if ((output[i] > 0 && output[i] < 1) &&
					(target[i] >= 0 && target[i] <= 1))
				{
					if (output[i] >= 0.5)
						output[i] = 1;
					else
						output[i] = 0;
				}

				if (output[i] == target[i])
					correct++;
			}
			all++;
			delete output;
			delete target;
		}
	}
	return eff = correct / all;
}

template<typename T>
void setrandomNumbers(std::string inFileName, std::string outFileName, int seed, T range)
{
	std::ifstream inFile(inFileName);
	std::ofstream outFile(outFileName);
	int layers;
	inFile >> layers;

	int type;
	inFile >> type;

	int* neurons = new int[layers];
	for (int i = 0; i < layers; i++)
	{
		inFile >> neurons[i];
	}

	outFile << layers << "\n";

	outFile << type << "\n";

	for (int i = 0; i < layers; i++)
	{
		outFile << neurons[i];
		if (i != layers - 1)
			outFile << " ";
	}
	outFile << "\n";

	for (int i = 0; i < layers; i++)
	{
		for (int j = 0; j < neurons[i]; j++)
		{
			if (i)
				outFile << randomNumber<T>(seed, range) << " ";
			else
				outFile << "0 ";
			seed++;
		}
		outFile << "\n";
	}

	for (int k = 0; k < layers - 1; k++)
	{
		for (int i = 0; i < neurons[k]; i++)
		{
			for (int j = 0; j < neurons[k + 1]; j++)
			{
				outFile << randomNumber<T>(seed, range) << " ";
				seed++;
			}
		}
		outFile << "\n";
	}
}

template<typename T>
void setrandomNumbers(NeuralNet<T>& net, int seed, T range)
{
	int layers = net.getLayersNum();
	for (int i = 0; i < layers; i++)
	{
		int neurons = net.getLayers()[i]->getNeuronsNum();
		for (int j = 0; j < neurons; j++)
		{
			if (i)
				net.getLayers()[i]->getNeurons()[j]->setBias(randomNumber<T>(seed, range));
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
				net.getMatrixes()[k]->setWeight(i, j, randomNumber<T>(seed, range));
				seed++;
			}
		}
	}
}