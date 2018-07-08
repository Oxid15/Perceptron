#include<math.h>
#include<fstream>
#include<cstdlib>
#include<iostream>

template<typename T>
T randomWeight(int seed)
{
	T weight;
	srand(seed);
	for (int i = 0; i < 16; i++)
	{
		weight = 0.0001*(rand() % 5000 + 3000);
	}
	return weight;
}

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
	int prevNum;
	int nextNum;
public:

	BaseNeuron()
	{
		prevNum = 1;
		nextNum = 1;
		in_x = new T;
		out_x = new T;
	}

	BaseNeuron(int _prevNum, int _nextNum)
	{
		prevNum = _prevNum;
		nextNum = _nextNum;

		if (_prevNum == 1)
		{
			in_x = new T;
			out_x = new T[_nextNum];
		}
		else
		{
			if (_nextNum == 1)
			{
				in_x = new T[_prevNum];
				out_x = new T;
			}
			else
			{
				in_x = new T[_prevNum];
				out_x = new T[_nextNum];
			}
		}
	}
						
	T* process(T* in_x, T* weights)
	{
		T _sum = sum<T>(in_x, prevNum);
		for (int i = 0; i < nextNum; i++)
		{
			out_x[i] = sig(_sum * weights[i]);
		}
		return out_x;
	}

	T getOutput(int index){return out_x[index];}

	int getNextNum(){return nextNum;}

	int getPrevNum(){return prevNum;}
};
									
template<typename T>
class MatrixNeuron
{
	BaseNeuron<T>* left;
	BaseNeuron<T>* right;
	T weight;
public:

	MatrixNeuron()
	{
		left = new BaseNeuron<T>;
		right = new BaseNeuron<T>;
		weight = 0;
	}

	void setLeft(BaseNeuron<T>* _neuron) { left = _neuron;}
	void setRight(BaseNeuron<T>* _neuron) { right = _neuron; }

	void setWeight(T _weight) {weight = _weight;}

	T getWeight() {return weight;}

	BaseNeuron<T>* getLeft() {return left;}
	BaseNeuron<T>* getRight() { return right; }
};

template<typename T>
class AdjMatrix
{
	MatrixNeuron<T> ** arr;
	int length;
	int height;
public:
	AdjMatrix(int _length, int _height, T* weights, BaseNeuron<T>** leftNeurons, BaseNeuron<T>** rightNeurons)
	{
		length = _length;
		height = _height;
		arr = new MatrixNeuron<T>*[length];
		for (int i = 0; i < length; i++)
		{
			arr[i] = new MatrixNeuron<T>[height];
		}

		int k = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < length; j++)
			{
				arr[j][i].setLeft(leftNeurons[k]);
				arr[j][i].setRight(rightNeurons[k]);
				arr[j][i].setWeight(weights[k]);
				k++;
			}
		}
	}

	AdjMatrix(int _length, int _height, T* weights)
	{
		length = _length;
		height = _height;
		arr = new MatrixNeuron<T>*[length];
		for (int i = 0; i < length; i++)
		{
			arr[i] = new MatrixNeuron<T>[height];
		}

		int k = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < length; j++)
			{
				arr[j][i].setLeft(new BaseNeuron<T>);
				arr[j][i].setRight(new BaseNeuron<T>);
				arr[j][i].setWeight(weights[k]);
				k++;
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
		MatrixNeuron<T> neuron = arr[i][j];
		return neuron.getWeight();
	}

	//T** getAllWeights()
	//{
	//	T** temp = new T*[height];
	//	for (int i = 0; i < height; i++)
	//	{
	//		for (int j = 0; j < length; j++)
	//		{
	//			temp[j][i] = arr[j][i].getWeight();
	//		}
	//	}
	//	return temp;
	//}

	int getLength() { return length; }
	int getHeight() { return height; }

	//void printWeights()
	//{
	//	for (int i = 0; i < height; i++)
	//	{
	//		for (int j = 0; j < length; j++)
	//		{
	//			std::cout << arr[j][i].getWeight();
	//			std::cout << ",";
	//		}
	//		std::cout << "\n";
	//	}
	//}
};

template<typename T>
class Layer
{
	BaseNeuron<T>** arr;
	T* output;
	int neurons;
	int lenInput;
	int lenOutput;
public:
	Layer(int quantityPrev, int _quantityOfNeurons, int quantityNext)
	{
		neurons = _quantityOfNeurons;
		arr = new BaseNeuron<T>*[neurons];
		for (int i = 0; i < neurons; i++)
		{
			arr[i] = new BaseNeuron<T>(quantityPrev, quantityNext);
		}

		lenInput = 0;
		int* prevNum = new int[neurons];
		for (int i = 0; i < neurons; i++)
		{
			prevNum[i] = arr[i]->getPrevNum();
			lenInput += prevNum[i];
		}

		lenOutput = 0;
		int* nextNum = new int[neurons];
		for (int i = 0; i < neurons; i++)
		{
			nextNum[i] = arr[i]->getNextNum();
			lenOutput += nextNum[i];
		}

		output = new T[lenOutput];
	}

	T* process(T* input, AdjMatrix<T>* matrix)
	{
		T** neuronInput = new T*[neurons];
		for (int i = 0; i < neurons; i++)
		{
			int prev = arr[i]->getPrevNum();
			neuronInput[i] = new T[prev];
		}

		int k = 0;
		for (int i = 0; i < neurons; i++)
		{
			int prev = arr[i]->getPrevNum();
			for (int j = 0; j < prev; j++, k++)
			{
				*neuronInput[j] = input[k];
			}
			arr[i]->process(neuronInput[i], matrix->getStrWeights(i));
		}
			
		//int k = 0;
		//for (int i = 0; i < neurons; i++)
		//{
		//	int prev = arr[i]->getPrevNum();
		//	T* neuronInput = new T[prev];
		//	for (int j = 0; j < prev; j++, k++)
		//	{
		//		neuronInput[j] = input[k];
		//	}
		//
		//	arr[i]->process(neuronInput, matrix->getStrWeights(i));
		//}

		T* out = new T[neurons];
		int p = 0;
		for (int i = 0; i < neurons; i++)
		{
			int next = arr[i]->getNextNum();
			for (int j = 0; j < next; j++,p++)
			{
				out[p] = arr[i]->getOutput(j);
			}
		}
		output = out;
		return out;
	}

	T* getOutput(){return output;}
	BaseNeuron<T>** getNeurons() { return arr; }
	int getQuantityOfNeurons() { return neurons; }
};

template<typename T>
class NeuralNet
{
	int layers;
	int matrixes;
	Layer<T>** arrLayers;
	AdjMatrix<T>** arrMatrixes;

	void getStrFromFile(std::ifstream& trainSet)
	{
				
	}

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
				arrLayers[i] = new Layer<T>(1, neurons[i], neurons[i + 1]);
				continue;
			}			 

			if (i == layers - 1)   
			{
				arrLayers[i] = new Layer<T>(neurons[i - 1], neurons[i], 1);
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
			arrMatrixes[i] = new AdjMatrix<T>(neurons[i + 1], neurons[i], weights[i], arrLayers[i]->getNeurons(), arrLayers[i + 1]->getNeurons());
		}

		for (int i = 0; i < matrixes; i++)
		{
			delete weights[i];
		}
		delete[] weights;
		delete[] neurons;
	}

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

		for (int i = 0; i < layers; i++)
		{
			if (i == 0)
			{
				arrLayers[i] = new Layer<T>(1, neurons[i], neurons[i + 1]);
				continue;
			}

			if (i == layers - 1)
			{
				arrLayers[i] = new Layer<T>(neurons[i - 1], neurons[i], 1);
				continue;
			}
			arrLayers[i] = new Layer<T>(neurons[i - 1], neurons[i], neurons[i + 1]);
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
		delete[] neurons;
	}

	void train(std::ifstream& trainSet, int numOfIterations)
	{
		
	}

	T* process(T* input)
	{
		arrLayers[0]->process(input, arrMatrixes[0]);

		for (int i = 1; i < layers - 1; i++)
		{
			arrLayers[i]->process(arrLayers[i - 1]->getOutput(), arrMatrixes[i]);
		}

		int size = arrLayers[layers-1]->getQuantityOfNeurons();
		T* weights = new T[size];
		for (int i = 0; i < size; i++)
		{
			weights[i] = 1;
		}
		AdjMatrix<T>* addMatrix = new AdjMatrix<T>(1,size,weights);
		arrLayers[layers-1]->process(arrLayers[layers-2]->getOutput(),addMatrix);
		return arrLayers[layers-1]->getOutput();			
	}

	void fileOutput(std::ofstream& file)
	{
		file << layers << "\n";
		
		for (int i = 0; i < layers; i++)
		{
			file << arrLayers[i]->getQuantityOfNeurons();
			if (i != layers -1) 
				file << " ";
		}
		file << "\n";

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
	std::cout << "sig(2*2) =" << sig<double>(2 * 2) << "\n";
	std::cout << "sig(2*4) =" << sig<double>(2 * 4) << "\n";
	std::cout << "sig(2*5) =" << sig<double>(2 * 5) << "\n";
	std::cout << "\n";
	std::cout << "sig(3*3) =" << sig<double>(3*3) << "\n";
	std::cout << "sig(3*6) =" << sig<double>(3*6) << "\n";
	std::cout << "sig(3*7) =" << sig<double>(3*7) << "\n";

	double y1 = sig<double>(2 * 2);
	double y2 = sig<double>(2 * 4);
	double y3 = sig<double>(2 * 5);
	double y4 = sig<double>(3 * 3);
	double y5 = sig<double>(3 * 6);
	double y6 = sig<double>(3 * 7);

	double z1 = sig<double>(2 * (y1 + y4));
	double z2 = sig<double>(3 * (y2 + y5));
	double z3 = sig<double>(4 * (y3 + y6));

	double c1 = sig<double>(z1 + z2 + z3);

	std::cout << "test output is: " << c1;

	std::ifstream file("mainConfig.txt");				   
	NeuralNet<double> n(file);
	std::ifstream trainFile("simpleTest.csv");

	double arr[2] = { 2,3 };
	std::cout << "\n";
	double* out = n.process(arr);
	std::cout << "net output is:" << out[0];
}