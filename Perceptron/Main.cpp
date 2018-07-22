#include<math.h>
#include<fstream>
#include<cstdlib>
#include<iostream>

template<typename T>
T randomWeight(int seed)
{
	T weight;
	srand(seed);
	weight = 0.0001*(rand() % 5000 + 3000);
	if (seed % 2)
		weight = weight - 2 * weight;
	return weight;
}

template<typename T>
T sig(T num) { return 1 / (1 + exp(-num)); }

template<typename T>
T sigDerivative(T num) { return sig<T>((num)*(1 - sig<T>(num))); }

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

enum functionType { sigmoid, step_func };

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
		if (type)
		{
			if (num > bias)
				return 1;
			else
				return 0;
		}
		else
			return sig(num);
	}

public:

	BaseNeuron()
	{
		prevNum = 1;
		nextNum = 1;
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
		out = activation(weighedSum(in, weights, prevNum), type);
		return out;
	}

	T process(T in)
	{
		out = in;
		return out;
	}

	void setBias(T error, T speed)
	{
		bias += speed * error;
	}

	T getBias() { return bias; }

	T getOutput() { return out; }

	int getNextNum() { return nextNum; }

	int getPrevNum() { return prevNum; }
};

template<typename T>
class MatrixUnit
{
	BaseNeuron<T>* leftNeuron;
	BaseNeuron<T>* rightNeuron;
	T weight;
public:

	MatrixUnit()
	{
		leftNeuron = new BaseNeuron<T>;
		rightNeuron = new BaseNeuron<T>;
		weight = 0;
	}

	~MatrixUnit()
	{
		delete leftNeuron;
		delete rightNeuron;
	}

	void setLeftNeuron(BaseNeuron<T>* _neuron) { leftNeuron = _neuron; }
	void setRightNeuron(BaseNeuron<T>* _neuron) { rightNeuron = _neuron; }

	void setWeight(T _weight) { weight = _weight; }

	void addToWeight(T addend)
	{
		weight += addend;							
	}

	T getWeight() { return weight; }

	BaseNeuron<T>* getleftNeuron() { return leftNeuron; }
	BaseNeuron<T>* getrightNeuron() { return rightNeuron; }
};

template<typename T>
class AdjMatrix
{
	MatrixUnit<T> ** arr;
	int length;
	int height;
public:
	AdjMatrix(int _length, int _height, T* weights, BaseNeuron<T>** leftNeurons, BaseNeuron<T>** rightNeurons)
	{
		length = _length;
		height = _height;
		arr = new MatrixUnit<T>*[length];
		for (int i = 0; i < length; i++)
		{
			arr[i] = new MatrixUnit<T>[height];
		}

		int k = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < length; j++)
			{
				arr[j][i].setLeftNeuron(leftNeurons[k]);
				arr[j][i].setRightNeuron(rightNeurons[k]);
				arr[j][i].setWeight(weights[k]);
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
				dWeights[j][i] = -(speed * layerInput[i] * layerError[j]);
			}
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < length; j++)
			{
				arr[j][i].addToWeight(dWeights[j][i]);
			}
		}

	}

	T* getStrWeights(int index)
	{
		T* temp = new T[length];
		for (int i = 0; i < length; i++)
		{
			temp[i] = arr[i][index].getWeight();
		}
		return temp;
	}

	T* getColWeights(int index)
	{
		T* temp = new T[height];
		for (int i = 0; i < height; i++)
		{
			temp[i] = arr[index][i].getWeight();
		}
		return temp;
	}

	T getWeight(int i, int j)
	{
		MatrixUnit<T> neuron = arr[i][j];
		return neuron.getWeight();
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

	void setError(T* target)
	{
		for (int i = 0; i < neurons; i++)
		{
			error[i] = (target[i] - output[i]) * sigDerivative<T>(sum<T>(input, lenInput) + arr[i]->getBias());
		}
	}

	void setError(T* errors, AdjMatrix<T>* matrix)
	{
		for (int i = 0; i < neurons; i++)
		{
			error[i] = weighedSum(errors, matrix->getStrWeights(i), neurons) * sigDerivative<T>(sum<T>(input, lenInput) + arr[i]->getBias());
		}
	}

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
		arrLayers[layers - 1]->setError(target);
		for (int i = layers - 2; i > 0; i--)
		{
			arrLayers[i]->setError(arrLayers[i + 1]->getError(), arrMatrixes[i]);
		}

		for (int i = 0; i < matrixes; i++)
		{
			arrMatrixes[i]->setWeights(arrLayers[i + 1]->getError(), arrLayers[i + 1]->getInput(), speed);
		}

		for (int i = 1; i < layers; i++)
		{
			for (int j = 0; j < arrLayers[i]->getNeuronsNum(); j++)
			{
				arrLayers[i]->getNeurons()[j]->setBias(arrLayers[i]->getError(j), speed);
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

	void writeLog(std::ofstream& logFile , std::ofstream& results, T* net_out, std::ifstream& set, int out_len)
	{
		for (int i = 0; i < out_len; i++)
		{
			if (net_out[i] >= 0.5)
				logFile << "1";
			else
				logFile << "0";
			logFile << ";";
		}	
		logFile << "\n";

		for (int i = 0; i < out_len; i++)
		{
			results << net_out[i];
			results << "\n";
		}
	}

public:
	NeuralNet(std::ifstream& configFile)
	{
		configFile >> layers;
		matrixes = layers - 1;

		arrLayers = new Layer<T>*[layers];
		arrMatrixes = new AdjMatrix<T>*[matrixes];

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

		int intType;
		functionType _type;
		configFile >> intType;
		if (intType)
			_type = step_func;
		else
			_type = sigmoid;

		type = _type;

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
			arrMatrixes[i] = new AdjMatrix<T>(neurons[i + 1], neurons[i], weights[i], arrLayers[i]->getNeurons(), arrLayers[i + 1]->getNeurons());
		}

		for (int i = 0; i < matrixes; i++)
		{
			delete weights[i];
		}
		delete[] weights;
		for (int i = 0; i < layers; i++)
		{
			delete biases[i];
		}
		delete[] biases;
		delete[] neurons;
	}


	void dataProcess(std::ifstream& set, int numOfIterations)
	{
		std::ofstream logFile("log.csv");
		std::ofstream results("log.txt");
		for (int i = 0; i < numOfIterations; i++)
		{
			int out_len = arrLayers[layers - 1]->getNeuronsNum();
			T* net_out = new T[out_len];
			net_out = process(getStrFromFile(set));
			writeLog(logFile, results, net_out, set, out_len);
		}
	}

	void train(std::ifstream& trainSet, int numOfIterations, T speed)
	{
		std::ofstream logFile("log.csv");
		std::ofstream logFile2("log.txt");
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

			writeLog(logFile, logFile2, net_out, trainSet, out_len);

			backpropagation(target_out, speed);						 
		}
	}

	T getEffiency(std::ifstream& logFile)
	{
		T all = 0;
		T right = 0;
		T effiency;
		int out_len = arrLayers[layers - 1]->getNeuronsNum();
		while (!logFile.eof())
		{
			T* output = new T[out_len];
			T* target = new T[out_len];

			for (int i = 0; i < out_len; i++)
			{
				logFile >> output[i];
				logFile.get();
			}

			for (int i = 0; i < out_len; i++)
			{
				logFile >> target[i];
				logFile.get();
			}

			for (int i = 0; i < out_len; i++)
			{
				if (output[i] = target[i])
					right++;
				all++;
			}
		}
		effiency = right / (all);
		return effiency;
	}

	void fileOutput(std::ofstream& file)
	{
		file << layers << "\n";

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

		if (type)
			file << "1\n";
		else
			file << "0\n";
		

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
};

void main()
{
	std::ifstream file("testConfig7.txt");
	NeuralNet<double> n(file);
	std::ifstream test("simpleTest.csv");
	n.dataProcess(test, 4);
}