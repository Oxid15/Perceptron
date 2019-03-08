#include"computations.h"

template<typename T>
class Neuron
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
		case softpls:
			return softplus(num);
		}
		return NULL;
	}

public:

	Neuron(int _prevNum = 1, int _nextNum = 1, T _bias = 0)
	{
		bias = _bias;
		prevNum = _prevNum;
		nextNum = _nextNum;
		in = new T[prevNum];
	}

	~Neuron() { delete in; }

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

	void addToBias(T error, T speed) { bias += speed * error; }

	void setBias(T _bias) { bias = _bias; }

	T getBias() { return bias; }

	T getOutput() { return out; }

	int getNextNum() { return nextNum; }

	int getPrevNum() { return prevNum; }
};

