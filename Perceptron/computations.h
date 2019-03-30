#include<math.h>									  
#include<time.h>
#include<random>

enum functionType { sigmoid, softpls, th};

template<typename T>
T randomNumber(int seed, std::default_random_engine& engine, int max, int min = 0)
{
	std::uniform_real_distribution<T> d(min, max);
	T weight;
	weight = d(engine);
	return weight;
}

template<typename T>
void swap(T& left, T& right)
{
	T tmp = left;
	left = right;
	right = tmp;
}

template<typename T>
T* elemPow(T* vect, int size, int power)
{
	for (int i = 0; i < size; i++)
	{
		vect[i] = pow(vect[i], power);
	}
	return vect;
}

template<typename T>
T tanh(T num) 
{
	T ex = exp(2 * num);
	return (ex - 1)/(ex + 1); 
}

template<typename T>
T sech(T num) 
{
	return 2 / (exp(-num) + exp(num));
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
T derivative(functionType ftype, T num)
{
	switch (ftype)
	{
	case sigmoid:
		return sigDerivative(num);
	case softpls:
		return softplusDerivative(num);
	case th:
		return pow(sech(num), 2);
	}
	return NULL;
}

template<typename T>
void insertionSort(T* arr, int size)
{
	T key = 0;
	int i = 0;
	for (int j = 1; j < size; j++) 
	{
		key = arr[j];
		i = j - 1;
		while (i >= 0 && arr[i] > key) 
		{
			arr[i + 1] = arr[i];
			i = i - 1;
			arr[i + 1] = key;
		}
	}
}

template<typename T>
T sum(T* arr, int n)
{
	T sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += arr[i];
	}
	return sum;
}

template<typename T>
T euclidNorm(T* vect, int size)
{
	return pow(sum<T>(elemPow<T>(vect, size, /*pow =*/2), size), 0.5);
}

template<typename T>
T euclidNorm(T* vect1, T* vect2, int size)
{
	T* arr = new T[size];
	for (int i = 0; i < size; i++)
	{
		arr[i] = (vect1[i] - vect2[i]);
	}
	T norm = pow((sum<T>(elemPow<T>(arr, size, 2), size)), 0.5);
	delete arr;
	return norm;
}

template<typename T>
T mean(T* arr, int n) { return sum<T>(arr, n) / n; }

template<typename T>
T* normalizeVect(T* vect, int size)
{
	T norm = euclidNorm<T>(vect, size);
	for (int i = 0; i < size; i++)
	{
		vect[i] /= norm;
	}
	return vect;
}

template<typename T>
T weighedSum(T* in, T* weights, int size)
{
	T sum = 0;
	for (int i = 0; i < size; i++)
	{
		sum += in[i] * weights[i];
	}
	return sum;
}

//returns the integer array where numbers is the quantity of values that is satisfying intervals
template<typename T>
void computeFrequencies(int* freq, T* arr, int size, int numOfIntervals)
{
	insertionSort<T>(arr, size);

	T max = arr[size - 1];
	T min = arr[0];

	T length = max - min;
	T dx = length / numOfIntervals;

	for (int i = 0; i < numOfIntervals; i++)
		freq[i] = 0;

	int bound = 0;
	T interval = 0;
	for (int i = 0; i < numOfIntervals; i++)
	{
		for (int j = bound; j < size; j++)
		{
			if (arr[j] >= interval && arr[j] < interval + dx)
			{
				freq[i]++;
				bound++;
			}
		}
	}
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
		size = newSize;
		//I have unexpected error when I try to delete previous array
		//it refers to the other classes logic
	}

public:

	expArray()
	{
		size = 2;
		arr = new T[size];
		cursor = 0;
	}

	expArray(T* data, int _size)
	{
		size = 2;
		arr = new T[size];
		cursor = 0;

		for (int i = 0; i < _size; i++)
		{
			this->add(data[i]);
		}
	}

	void add(T& data)
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
			this->expand();
			add(data, index);
		}
		else if (index >= 0 and index < cursor)
		{
			if (cursor == size)
				expand();

			for (int i = cursor - 1; i >= index; i--)
			{
				arr[i + 1] = arr[i];
			}
			arr[index] = data;
			cursor++;
		}
		else
		{
			arr[cursor] = data;
			cursor++;
		}
	}

	void del(int index = 0)
	{
		try
		{
			if (index < size)
			{
				int j = 0;
				T* newArr = new T[size];
				for (int i = 0; i < cursor; i++)
				{
					if (i != index)
					{
						newArr[j] = arr[i];
						j++;
					}
				}
				arr = newArr;
				cursor = index;
			}
			else
				throw "Access violation";
		}
		catch (char* str)
		{
			std::cerr << str << "\n";
		}
		return;
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
				throw "Access violation";
		}
		catch (char* str)
		{
			std::cerr << str << "\n";
		}
		return nullptr;
	}

	int getSize() { return size; }
};