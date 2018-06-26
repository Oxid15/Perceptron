#include<math.h>		   //exp

template<typename T>
T sig(T num) { return 1 / (1 + exp(-num)); }

template<typename T>
T sum(T* in_x, int n)
{
	int sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += in_x[i];
	}
	return sum;
}

template<typename T>
class BaseNeuron
{
public:
	struct WeighedNeuron
	{
		BaseNeuron* neuron;
		T weight;
	};

	T* in_x;
	T* out_x;
	int nextNum;
	int prevNum;
	WeighedNeuron** next;
	BaseNeuron** prev;

	BaseNeuron(int _nextNum, int _prevNum)
	{
		nextNum = _nextNum;
		prevNum = _prevNum;
		in_x = new T[prevNum];
		out_x = new T[nextNum];
		next = new WeighedNeuron*[nextNum];
		prev = new BaseNeuron*[prevNum];
	}
						
	void process(T* in_x)
	{
		T _sum = sum<T>(in_x, prevNum);
		for (int i = 0; i < nextNum; i++)
		{
			out_x[i] = sig(_sum * next[i].weight);
		}
	}

	void setConnection(BaseNeuron* nextN, T weight, int i, int j)
	{
		WeighedNeuron* newN = new WeighedNeuron;
		newN->neuron = nextN;
		newN->weight = weight;
		next[i] = newN;

		nextN->prev[j] = this;
	}

	void setError()
	{
		
	}
};

template<typename T>
class Layer
{
	BaseNeuron** arr;
	int quantity;
	Layer(int _quantity)
	{
		quantity = _quantity;
		arr = new BaseNeuron[quantity];
	}
};

int main()
{
	return 0;	   
}																			   