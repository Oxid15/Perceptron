#include<math.h>
#include<fstream>									  
#include<cstdlib>
#include<iostream>
#include<string>
#include<time.h>

enum functionType { sigmoid, softpls };

template<typename T>
T getValueFromCsv(std::fstream& file)
{
	T* output = new T;
	file << *output;
	file.get();
	return *output;
}

template<typename T>
T* getStrFromFile(std::ifstream& trainSet)
{
	int size = arrLayers[0]->getNeuronsNum();
	T* output = new T[size];
	for (int i = 0; i < size; i++)
	{
		output[i] = getValueFromCsv(trainSet);
	}
	return output;
}

template<typename T>
void setrandomWeights(std::string inFileName, std::string outFileName, int seed, T range)
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
				output[i] = getValueFromCsv(file);
			}
			for (int i = 0; i < output_length; i++)
			{
				output[i] = getValueFromCsv(file);
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
void setrandomWeights(NeuralNet<T>& net, int seed, T range)
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

template<typename T>
T randomNumber(int seed, int range)
{
	T weight;
	srand(seed);
	weight = (rand() % range + rand()*0.00001);
	if (seed % 2)
		weight = -weight;
	return weight;
}

template<typename T>
T sig(T num) { return 1 / (1 + exp(-num)); }

template<typename T>
T sigDerivative(T num)
{
	T ex = exp(-num);
	return ex / ((1 + ex)*(1 + ex));
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
class expArray
{
	T* arr;
	int size;
	int cursor;

	void expand()
	{
		int newSize = this->size * 2;
		T* newArr = new T[newSize];
		for (int i = 0; i < size; i++)
		{
			newArr[i] = arr[i];
		}
		arr = newArr;
		size *= 2;
	}

public:

	expArray()
	{
		arr = new T[2];
		size = 2;
		cursor = 0;
	}

	void add(T data)
	{
		if (cursor == size)
		{
			expand();
			add(data);
			return;
		}
		else
		{
			arr[cursor] = data;
			cursor++;
		}
	}

	void add(T& data, int index)
	{
		if (index == size)
		{
			expand();
			add(data, index);
			return;
		}
		else if (index < size and index >= 0)
		{
			arr[index] = data;
			cursor++;
		}
		else
		{
			arr[cursor] = data;
			cursor ++;
		}
	}

	T* operator [] (int n)
	{
		try
		{
			if (n < size)
			{
				return &arr[n];
			}
			else
				throw "Error: access violation";
		}
		catch (char* str)
		{
			std::cerr << str << "\n";
		}
		return nullptr;
	}

	T* getArr()
	{
		return arr;
	}
};