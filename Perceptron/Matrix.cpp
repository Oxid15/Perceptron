#include "Neuron.cpp"
#include"files.h"

template<typename T>
class Matrix
{
	T** arr;
	int length;
	int height;

public:
	Matrix()
	{
		length = 1;
		height = 1;
		arr = new T*[length];
		for (int i = 0; i < length; i++)
		{
			arr[i] = new T[height];
		}
	}

	Matrix(int _length, int _height, int seed, T maxWeight, T minWeight)
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
				arr[j][i] = unifRealRandNum<T>(seed, engine, maxWeight, minWeight);
				k++;
			}
		}
	}

	Matrix(T* weights, int _length, int _height)
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

	~Matrix()
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
		try
		{
			if (index < height)
			{
				//this is convenient, but may cause memory overflow
				T* temp = new T[length];
				for (int i = 0; i < length; i++)
				{
					temp[i] = arr[i][index];
				}
				return temp;
			}
			else
				throw "Access violation!";
		}

		catch (char* str)
		{
			std::cerr << str << "\n";
		}

		return nullptr;
	}

	T* getColWeights(int index)
	{
		try
		{
			if (index < length)
			{
				//same
				T* temp = new T[height];
				for (int i = 0; i < height; i++)
				{
					temp[i] = arr[index][i];
				}
				return temp;
			}
			else
				throw "Access violation!";
		}
		catch (char* str)
		{
			std::cerr << str << "\n";
		}

		return nullptr;
	}

	void setWeight(int i, int j, T weight)
	{
		try
		{
			if (i < length && j < height)
				arr[i][j] = weight;
			else
				throw "Access violation!";
		}
		catch (char* str)
		{
			std::cerr << str << "\n";
		}
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

		for (int i = 0; i < length; i++)
			delete dWeights[i];
		delete dWeights;
	}

	T getWeight(int i, int j)
	{
		try
		{
			if (i < length && j < height)
				return arr[i][j];
			else
				throw "Access violation!";
		}
		catch (char* str)
		{
			std::cerr << str << "\n";
		}							
	}

	int getLength() { return length; }
	int getHeight() { return height; }
};