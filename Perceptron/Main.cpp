#include<math.h>
#include<fstream>									  
#include<cstdlib>
#include<iostream>
#include<string>

enum functionType { sigmoid, treshold_func, relu, softpls };

template<typename T>
T randomWeight(int seed)
{
	T weight;
	srand(seed);
	weight = (rand() % 20);
	if (seed % 2)
		weight = weight - 2 * weight;
	return weight;
}

template<typename T>
void setRandomWeights(std::string inFileName, std::string outFileName, int seed)
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
				outFile << randomWeight<T>(seed) << " ";
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
				outFile << randomWeight<T>(seed) << " ";
				seed++;
			}
		}
		outFile << "\n";
	}
}

template<typename T>
T sig(T num) { return 1 / (1 + exp(-num)); }

template<typename T>
T sigDerivative(T num) { return sig<T>((num)*(1 - sig<T>(num))); }

template<typename T>
T treshold_function(T num)
{
	if (num >= 0)
		return 1;
	else
		return 0;
}

template<typename T>
T ReLU(T num)
{
	if (num >= 0)
		return num;
	else
		return 0;
}

template<typename T>
T softplus(T num) { return log(1 + exp(num)); }

template<typename T>
T softplusDerivative(T num) { return sig(num); }

template<typename T>
T derivative(functionType type, T num)
{
	switch (type)
	{
	case sigmoid:
		return sigDerivative(num);
	case softpls:
		return softplusDerivative(num);
	}
}


template<typename T>
T sum(T* in, int n)
{
	T sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += in[i];
	}
	return sum;
}

template<typename T>
T weighedSum(T* in, T* weights, int n)
{
	T sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += in[i] * weights[i];
	}
	return sum;
}

template<typename T>
T getEffiency(std::string fileName, int output_length, int numberOfPairs)
{
	std::ifstream file(fileName);
	T all = 0;
	T correct = 0;
	T eff;
	for (int p = 0; p < numberOfPairs; p++)
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

void clearFiles(std::string* names, int number)
{
	for (int i = 0; i < number; i++)
	{
		std::ofstream file(names[i]);
		file << '\b';
	}
}

template<typename T>
class BaseNeuron
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
		case treshold_func:
			return treshold_function(num);
		case relu:
			return ReLU(num);
		case softpls:
			return softplus(num);
		}
	}

public:

	BaseNeuron()
	{
		bias = 0;
		prevNum = 1;
		nextNum = 1;
		in = new T[prevNum];
	}

	~BaseNeuron() { delete in; }

	BaseNeuron(int _prevNum, int _nextNum, T _bias)
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
public:
	AdjMatrix(int _length, int _height, T* weights)
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
				dWeights[j][i] = speed * layerInput[i] * layerError[j];
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
	BaseNeuron<T>** arr;
	T* input;
	T* output;
	T* error;
	int neurons;
	int lenInput;
public:
	Layer(int prevNum, int _neuronsNum, int nextNum, T* biases)
	{
		neurons = _neuronsNum;
		arr = new BaseNeuron<T>*[neurons];
		for (int i = 0; i < neurons; i++)
		{
			arr[i] = new BaseNeuron<T>(prevNum, nextNum, biases[i]);
		}

		lenInput = prevNum;

		input = new T[lenInput];
		output = new T[neurons];
		error = new T[neurons];
	}

	~Layer()
	{
		for (int i = 0; i < neurons; i++)
		{
			delete arr[i];
		}
		delete[] arr;
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

	void setError(T* target, functionType type)
	{
		for (int i = 0; i < neurons; i++)
		{
			error[i] = (target[i] - output[i]) * derivative<T>(type, sum<T>(input, lenInput) + arr[i]->getBias());
		}
	}

	void setError(T* errors, AdjMatrix<T>* matrix, functionType type)
	{
		for (int i = 0; i < neurons; i++)
		{
			error[i] = weighedSum(errors, matrix->getStrWeights(i), arr[i]->getNextNum()) * 
				derivative<T>(type, sum<T>(input, lenInput) + arr[i]->getBias());		  
		}
	}

	void setNeurons(int _neurons) { neurons = _neurons; }

	T* getInput() { return input; }

	T* getOutput() { return output; }

	T* getError() { return error; }

	T getError(int index) { return error[index]; }

	BaseNeuron<T>** getNeurons() { return arr; }

	int getNeuronsNum() { return neurons; }
};

template<typename T>
class NeuralNet
{
	int layers;
	int matrixes;
	Layer<T>** arrLayers;
	AdjMatrix<T>** arrMatrixes;
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
		if (type == sigmoid || type == softpls)
		{
			arrLayers[layers - 1]->setError(target, type);
			for (int i = layers - 2; i >= 0; i--)
			{
				arrLayers[i]->setError(arrLayers[i + 1]->getError(), arrMatrixes[i], type);
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
		else
		{
			// delta-rule if layers == 2
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
	NeuralNet(std::string fileName)
	{
		std::ifstream configFile(fileName);
		configFile >> layers;
		matrixes = layers - 1;

		arrLayers = new Layer<T>*[layers];
		arrMatrixes = new AdjMatrix<T>*[matrixes];

		int _type;
		configFile >> _type;
		switch (_type)
		{
		case 0:
			type = sigmoid;
			break;
		case 1:
			type = treshold_func;
			break;
		case 2:
			type = relu;
			break;
		case 3:
			type = softpls;
		}

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

		for (int i = 0; i < layers; i++)
		{
			if (i == 0)
			{
				arrLayers[i] = new Layer<T>(1, neurons[i], neurons[i + 1], biases[i]);
				continue;
			}

			if (i == layers - 1)
			{
				arrLayers[i] = new Layer<T>(neurons[i - 1], neurons[i], 1, biases[i]);
				continue;
			}
			arrLayers[i] = new Layer<T>(neurons[i - 1], neurons[i], neurons[i + 1], biases[i]);
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

		for (int i = 0; i < matrixes; i++)
		{
			arrMatrixes[i] = new AdjMatrix<T>(neurons[i + 1], neurons[i], weights[i]);
		}
	}

	void dataProcess(std::string fileName, int numOfIterations)
	{
		std::ifstream set(fileName);
		std::ofstream log("testLog.csv");
		for (int i = 0; i < numOfIterations; i++)
		{
			int out_len = arrLayers[layers - 1]->getNeuronsNum();
			T* net_out = new T[out_len];
			net_out = process(getStrFromFile(set));

			T* target_out = new T[out_len];
			for (int i = 0; i < out_len; i++)
			{
				set >> target_out[i];
				set.get();
			}
			writeLog(net_out, target_out, log);
		}
	}

	void train(std::string fileName, int numOfIterations, T speed)
	{
		std::ifstream trainSet(fileName);
		std::ofstream exLog("excelLog.csv", std::ios::app);
		for (int i = 0; i < numOfIterations; i++)
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

			std::ofstream log("trainLog.csv", std::ios::app);
			writeLog(net_out, target_out, log);

			
			int len = arrLayers[layers - 1]->getNeuronsNum();
			if (len != 1)
			{
				// complex output
			}
			else
			{
				exLog << net_out[0] << ";";
			}

		}
		exLog << "\n";
	}

	void fileOutput(std::string fileName)
	{
		std::ofstream file(fileName);
		file << layers << "\n";

		switch (type)
		{
		case sigmoid:
			file << "0\n";
			break;
		case treshold_func:
			file << "1\n";
			break;
		case relu:
			file << "2\n";
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
				file << arrLayers[i]->getNeurons()[j]->getBias() << " ";
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
					file << arrMatrixes[k]->getWeight(j, i) << " ";
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
				file << arrLayers[i]->getNeurons()[j]->getBias() << ";";
			}
		}

		for (int i = 0; i < matrixes; i++)
		{
			arrMatrixes[i]->fileOutput(file);
		}
		file << "\n";
	}
};

void main()
{
	std::string logs[5] = { "testLog.csv", "effLog.csv","trainLog.csv","weightLog.csv","excelLog.csv" };
	clearFiles(logs,5);

	int num = 1;
	for (int k = 0; k < num; k++)
	{
		setRandomWeights<double>("testConfig.txt", "currentConfig.txt", k*k+k);

		int numEpoch = 1000;
		int numIter = 4;
		for (int i = 0; i < numEpoch; i++)
		{
			NeuralNet<double>n("currentConfig.txt");

			n.train("simpleTrain.csv", numIter, 1);

			n.dataProcess("simpleTest.csv", numIter);

			std::ofstream effLog("effLog.csv", std::ios::app);
			double eff = getEffiency<double>("testLog.csv", 1, numIter);
			effLog << eff << ";\n";
			effLog.close();

			n.fileOutput("currentConfig.txt");
			n.weightsOutput("weightLog.csv");

			if (eff == 1)
				return;
		}
	}
}