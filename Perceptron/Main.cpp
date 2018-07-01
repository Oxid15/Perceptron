#include<math.h>
#include<iostream>

template<typename T>
T getValue()
{
	std::cout << "\ntype the value\n";
	T value;
	std::cin >> value;
	return value;
}

template<typename T>
T* getArray(int size)
{
	std::cout << "\ntype the array through the gaps\n";
	T* arr = new T[size];
	for (int i = 0; i < size; i++)
	{
		std::cin >> arr[i];
	}
	return arr;
}

template<typename T>
T sig(T num) { return 1 / (1 + exp(-num)); }

template<typename T>
T sum(T* in_x, int n)
{
	T sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += in_x[i];
	}
	return sum;
}

template<typename T>
class BaseNeuron
{
	T* in_x;
	T* out_x;
public:
	BaseNeuron(int prevNum, int nextNum)
	{
		if (prevNum == 0)
		{
			in_x = nullptr;
			out_x = new T[nextNum];
		}
		else
		{
			if (nextNum == 0)
			{
				in_x = new T[prevNum];
				out_x = nullptr;
			}
			else
			{
				in_x = new T[prevNum];
				out_x = new T[nextNum];
			}
		}
	}
						
	T* process(T* in_x, int* weights, int prevNum, int nextNum)
	{
		T _sum = sum<T>(in_x, prevNum);
		for (int i = 0; i < nextNum; i++)
		{
			out_x[i] = sig(_sum * weights[i]);
		}
		return out_x;
	}
};

template<typename T>
class WeighedNeuron
{
	BaseNeuron<T>* neuron;
	T weight;
public:

	void setNeuron(BaseNeuron<T>* _neuron)
	{
		neuron = _neuron;
	}

	void setWeight(T _weight)
	{
		weight = _weight;
	}

	T getWeight()
	{
		return weight;
	}

	BaseNeuron<T>* getNeuron()
	{
		return neuron;
	}
};

template<typename T>
class AdjMatrix
{
	WeighedNeuron<T> ** arr;
	int length;
	int height;
public:
	AdjMatrix(int _length, int _height, T* weights, BaseNeuron<T>** neurons)
	{
		length = _length;
		height = _height;
		arr = new WeighedNeuron<T>*[length];
		for (int i = 0; i < length; i++)
		{
			arr[i] = new WeighedNeuron<T>[height];
		}

		int k = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < length; j++)
			{
				arr[i][j].setWeight(weights[k]);
				arr[i][j].setNeuron(neurons[k]);
				k++;
			}
		}
	}

	T* getStrWeights(int index)
	{
		T* temp = new T[length];
		for (int i = 0; i < length; i++)
		{
			temp[i] = arr[index][i].getWeight();
		}
		return temp;
	}

	T* getColWeights(int index)
	{
		T* temp = new T[height];
		for (int i = 0; i < height; i++)
		{
			temp[i] = arr[i][index].getWeight();
		}
		return temp;
	}

	void printWeights()
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < length; j++)
			{
				std::cout << arr[i][j].getWeight();
				std::cout << " ";
			}
			std::cout << "\n";
		}
	}
};

template<typename T>
class Layer
{
	BaseNeuron<T>** arr;
	int neurons;
public:
	Layer(int quantityPrev, int _quantityOfNeurons, int quantityNext)
	{
		neurons = _quantityOfNeurons;
		arr = new BaseNeuron<T>*[neurons];

		for (int i = 0; i < neurons; i++)
		{
			arr[i] = new BaseNeuron<T>(quantityPrev, quantityNext);
		}
	}

	BaseNeuron<T>** getNeurons()
	{
		return arr;
	}
};

template<typename T>
class NeuralNet
{
	int layers;
	int matrixes;
	Layer<T>** arrLayers;
	AdjMatrix<T>** arrMatrixes;
public:																
	NeuralNet(int _quantityOfLayers)
	{
		layers = _quantityOfLayers;
		matrixes = layers - 1;
						   											
		arrLayers = new Layer<T>*[layers];
		arrMatrixes = new AdjMatrix<T>*[matrixes];

		int* neurons = new int[layers];
		for (int i = 0; i < layers; i++)
		{
			neurons[i] = getValue<int>();
		}

		for (int i = 0; i < layers; i++)
		{
			if (i == 0)
			{
				arrLayers[i] = new Layer<T>(0, neurons[i], neurons[i + 1]);
				continue;
			}			 

			if (i == layers - 1)   
			{
				arrLayers[i] = new Layer<T>(neurons[i - 1], neurons[i], 0);
				continue;
			}
			arrLayers[i] = new Layer<T>(neurons[i - 1], neurons[i], neurons[i + 1]);
		}

		T** weights = new T*[matrixes];
		for (int i = 0; i < matrixes; i++)
		{
			weights[i] = getArray<T>(neurons[i] * neurons[i + 1]);
		}

		for (int i = 0; i < matrixes; i++)
		{
			arrMatrixes[i] = new AdjMatrix<T>(neurons[i + 1], neurons[i], weights[i], arrLayers[i + 1]->getNeurons());
		}

		for (int i = 0; i < matrixes; i++)
		{
			delete weights[i];
		}
		delete[] weights;
		delete[] neurons;
	}
};

void main()
{
	NeuralNet<double> net(2);
}

//3 4 1 2 3 4 5 6 7 8 9 10 11 12