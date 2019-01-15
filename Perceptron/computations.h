#include<math.h>									  
#include<time.h>

enum functionType { sigmoid, softpls };

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
T euclidNorm(T* vect1, T* vect2, int size)
{
	T* arr = new T[size];
	for (int i = 0; i < size; i++)
	{
		arr[i] = (vect1[i] - vect2[i])*(vect1[i] - vect2[i]);
	}
	return pow(sum<T>(arr, size), 0.5);
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
T mean(T* arr, int n) { return sum<T>(arr, n) / n; }

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
			cursor++;
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