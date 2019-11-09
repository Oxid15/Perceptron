#include<math.h>									  
#include<time.h>
#include<random>

enum functionType { sigmoid, softpls, th};

template<typename T>
T unifRealRandNum(int seed, std::default_random_engine& randEngine, int max, int min = 0)
{
	std::uniform_real_distribution<T> dist(min, max);
	T num = dist(randEngine);
	return num;
}

template<typename T>
T max(T* arr, int size)
{
	T max = arr[0];
	for (int i = 0; i < size; i++)
	{
		if (arr[i] > max)
			max = arr[i];
	}
	return max;
}

template<typename T>
T min(T* arr, int size)
{
	T min = arr[0];
	for (int i = 0; i < size; i++)
	{
		if (arr[i] < min)
			min = arr[i];
	}
	return min;
}

template<typename T>
T* elemPow(T* vect, int size, T power)
{
	for (int i = 0; i < size; i++)
		vect[i] = pow(vect[i], power);
	return vect;
}

template<typename T>
T tanh(T num) 
{
	T ex = exp(2 * num);
	return (ex - 1)/(ex + 1); 
}

template<typename T>
T sech(T num) { return 2 / (exp(-num) + exp(num)); }

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
T LpNorm(T* vect, int size, int p)
{
	//if p is even then we don't need abs
	if (p % 2 != 0)
		for (int i = 0; i < size; i++)
			vect[i] = abs(vect[i]);
	return pow( sum<T>( elemPow<T>(vect, size, p), size ), 1./p );
}

//returns euclidean distance between two vectors
//with the same size
template<typename T>
T euclidDist(T* vect1, T* vect2, int size)
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
T mean(T* arr, int size) { return sum<T>(arr, size) / size; }

template<typename T>
T median(T* arr, int size)
{
	insertionSort(arr, size);
	size % 2 != 0 ? return arr[(size + 1) / 2 - 1]: return (arr[size / 2 - 1] + arr[size / 2]) / 2;
}

template<typename T>
T variance(T* arr, int size, bool isShifted) 
{ 
	T mx = mean(arr, size);
	T total = 0;
	for (int i = 0; i < size; i++)
	{
		total += (arr[i] - mx) * (arr[i] - mx);
	}
	isShifted ? return total / size : return total / (size - 1);
}

//standard deviation
template<typename T>
T SD(T* arr, int size, bool isShifted)
{
	if(isShifted)
		return sqrtl(variance(arr, size, false));
	else
	{
		double n = (double)size;
		T corrCoeff = 1 + 1 / (4 * n) + 9 / (32 * n*n);
		return sqrtl(variance(arr, n, false)) * corrCoeff;
	}
}

//robust measure of scale
//MAD (median absolute deviation) * 1.4826
template<typename T>
T MeasOfScale(T* arr, int size)
{
	T med = median(arr, size);
	T* MAD = new T[size];
	for (int i = 0; i < size; i++)
	{
		MAD[i] = abs(arr[i] - med);
	}
	T resMedian = median(MAD, size) * 1.4826;
	delete MAD;

	return resMedian;
}

template<typename T>
T rawKthMoment(T* arr,int k, int size)
{
	if (k == 1)
		return mean(arr, size);
	else
	{
		T* total = new T[size];
		for (int i = 0; i < size; i++)
		{
			total[i] = pow(arr[i], k);
		}
		T moment = sum(total, size) / double(size);
		delete total;

		return moment;
	}
}

template<typename T>
T centralKthMoment(T* arr, int k, int size)
{
	T mx = mean(arr, size);
	T* total = new T[size];
	for (int i = 0; i < size; i++)
	{
		total[i] = pow((arr[i] - mx), k);
	}
	T moment = sum(total, size) / double(size);
	delete total;

	return moment;
}

//normalizes vector values by dividing them by 
//euclidean norm of this vector
template<typename T>
T* normalizeVect(T* vect, int size)			//TODO: make (an overloaded) 				
{											//function for another normalization methods (AND for matrices)
	T norm = LpNorm<T>(vect, size,/*pow=*/2);
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
	for (int i = 0; i < size; i++) sum += in[i] * weights[i];
	return sum;
}

//returns the integer array where numbers is 
//the quantity of values that is satisfying given intervals
template<typename T>
void computeFrequencies(int* result, T* arr, int size, int numOfIntervals)
{
	insertionSort<T>(arr, size);
	for (int i = 0; i < numOfIntervals; i++)
		result[i] = 0;

	T min = arr[0];
	T max = arr[size - 1];
	T dx = (max - min) / numOfIntervals; //lenght of one interval

	T interval = min;
	int bound = 0;
	for (int i = 0; i < numOfIntervals; i++)
	{
		for (int j = bound; j < size; j++)
		{
			if (arr[j] >= interval && arr[j] < interval + dx)
			{
				T left = interval;
				T right = interval + dx;
				result[i]++;
				bound = j; //to not to check elements that have already been checked
			}
			//includes last value that is equal to right limit
			if (arr[j] == interval + dx && j == size - 1)
			{
				result[i]++;
				bound = j;
			}
		}
		interval += dx;
	}
}

//computes cumulative distribution function
template<typename T>
void computeCDF(T* CDF, T* arr, int size, int numOfIntervals)
{
	int* freq = new int[numOfIntervals];
	computeFrequencies<T>(freq, arr, size, numOfIntervals);

	for (int i = 0; i < numOfIntervals; i++)
		CDF[i] = 0;

	CDF[0] = (double)freq[0] / size;
	for (int i = 1; i < numOfIntervals; i++)
	{
		CDF[i] += CDF[i - 1] + (double)freq[i] / size;
	}
	delete freq;
}

template<typename T>
void computePDF(T* PDF, T* arr, int size, int numOfIntervals)
{
	int* freq = new int[numOfIntervals];
	computeFrequencies(freq, arr, size, numOfIntervals);

	T min = arr[0];
	T max = arr[size - 1];
	T dx = (max - min) / numOfIntervals;

	for (int i = 0; i < numOfIntervals; i++)	//using max/min is necessary because																	//I need to keep the order of denseFunc	safe
		PDF[i] = freq[i] / (dx * size);			//therefore I cannot use insertionSort()
	delete freq;
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
		//it refers to the logic of the other classes
		//it works with simple types and classes but not with ones in this program
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
