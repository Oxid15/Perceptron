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
T* elemPow(T* vect, int size, int power)
{
	T* arr = new T[size];
	for (int i = 0; i < size; i++)
	{
		arr[i] = pow(vect[i], power);
	}
	return arr;
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
T derivative(functionType type, T num)
{
	switch (type)
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
T euclidNorm(T* vect, int size)
{
	return pow(sum<T>(elemPow<T>(vect, size, 2), size), 0.5);
}

template<typename T>
T euclidNorm(T* vect1, T* vect2, int size)
{
	T* arr = new T[size];
	for (int i = 0; i < size; i++)
	{
		arr[i] = (vect1[i] - vect2[i]);
	}
	return pow((sum<T>(elemPow<T>(arr, size, 2), size)), 0.5);
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

	T* getArr()
	{
		return arr;
	}
};