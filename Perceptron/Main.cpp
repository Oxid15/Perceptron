#include<math.h>
#include<fstream>									  
#include<cstdlib>
#include<iostream>
#include<string>

template<typename T>
T randomWeight(int seed)
{
	T weight;
	srand(seed);
	weight = 0.1*(rand() % 100);
	if (seed % 2)
		weight = weight - 2 * weight;
	return weight;
}

template<typename T>
void setRandomWeights(std::ifstream& inFile, std::ofstream& outFile, int seed)
{
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
T getEffiency(std::ifstream& file, int output_length, int numberOfPairs)
{
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

void clearFiles(std::string* name, int number)
{
	for (int i = 0; i < number; i++)
	{
		std::ofstream file(name[i]);
		file << '\b';
	}
}

enum functionType { sigmoid, treshold_func, relu, softpls };

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
				if (j != length - 1)
				{
					arr[j][i].setLeftNeuron(leftNeurons[k]);
					arr[j][i].setRightNeuron(rightNeurons[k]);
					arr[j][i].setWeight(weights[k]);
					k++;
				}
				else
				{
					arr[j][i].setLeftNeuron(leftNeurons[k]);
					arr[j][i].setRightNeuron(new BaseNeuron<T>);
					arr[j][i].setWeight(weights[k]);
					k++;
				}
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
				dWeights[j][i] = (speed * layerInput[i] * layerError[j]);				////////--------
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

	void fileOutput(std::ofstream& file)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < length; j++)
			{
				file << arr[j][i].getWeight() << ";";
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
			error[i] = (target[i] - output[i]) * sigDerivative<T>(sum<T>(input, lenInput) + arr[i]->getBias());					///////////////////
		}
	}

	void setError(T* errors, AdjMatrix<T>* matrix)
	{
		for (int i = 0; i < neurons; i++)
		{
			error[i] = weighedSum(errors, matrix->getStrWeights(i), neurons) * sigDerivative<T>(sum<T>(input, lenInput) + arr[i]->getBias());		  /////////////////
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
		if (type == sigmoid || type == softpls)
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
	NeuralNet(std::ifstream& configFile)
	{
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
			arrMatrixes[i] = new AdjMatrix<T>(neurons[i + 1], neurons[i], weights[i], arrLayers[i]->getNeurons(), arrLayers[i + 1]->getNeurons());
		}

		for (int i = 0; i < matrixes; i++){ delete weights[i]; }
		delete[] weights;
		for (int i = 0; i < layers; i++) { delete biases[i]; }
		delete[] biases;
		delete[] neurons;
	}

	void dataProcess(std::ifstream& set, int numOfIterations)
	{
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

	void train(std::ifstream& trainSet, int numOfIterations, T speed)
	{
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
		}
	}

	void fileOutput(std::ofstream& file)
	{
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

	void weightsOutput(std::ofstream& file)
	{
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
	int numEpoch = 12000;
	int numIter = 4;
	for (int i = 0; i < numEpoch; i++)
	{
		std::ifstream config("currentConfig.txt");
		NeuralNet<float>n(config);
		config.close();

		std::ifstream set("simpleTest.csv");
		n.train(set, numIter, 0.001);
		set.close();

		std::ifstream test("simpleTest.csv");
		n.dataProcess(test, numIter);
		test.close();

		std::ifstream testLog("testLog.csv");
		std::ofstream effLog("effLog.csv", std::ios::app);
		effLog << getEffiency<float>(testLog, 1, numIter) << ";\n";
		testLog.close();
		effLog.close();

		std::ofstream output("currentConfig.txt");
		n.fileOutput(output);
		output.close();

		std::ofstream weightLog("weightLog.csv", std::ios::app);
		n.weightsOutput(weightLog);
		weightLog.close();
	}

	//std::string logs[3] = { "testLog.csv", "effLog.csv","trainLog.csv" };
	//clearFiles(logs,3);
}