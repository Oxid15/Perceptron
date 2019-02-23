#include "Matrix.cpp"

template<typename T>
class Layer
{
	expArray<Neuron<T>> arr;
	T* input;
	T* output;
	T* error;
	int neurons;
	int prevNum;
	int nextNum;
public:
	Layer()
	{
		input = new T;
		output = new T;
		error = new T;
		neurons = 1;
		prevNum = 1;
		nextNum = 1;
	}

	Layer(T* biases, int _prevNum = 1, int _neuronsNum = 1, int _nextNum = 1)
	{
		neurons = _neuronsNum;
		prevNum = _prevNum;
		nextNum = _nextNum;
		input = new T[prevNum];
		output = new T[neurons];
		error = new T[neurons];

		for (int i = 0; i < neurons; i++)
			arr.add(*new Neuron<T>(prevNum, _nextNum, biases[i]));
	}

	~Layer()
	{
		delete input;
		delete output;
		delete error;
	}

	T* process(T* _input)
	{
		input = _input;
		for (int i = 0; i < neurons; i++)
		{
			output[i] = arr[i]->process(_input[i]);
		}
		return output;
	}

	T* process(T* _input, Matrix<T>* matrix, functionType type)
	{
		input = _input;
		for (int i = 0; i < neurons; i++)
		{
			output[i] = arr[i]->process(_input, matrix->getColWeights(i), type);
		}
		return output;
	}

	void setError(T* target, Matrix<T>* matrix, functionType funcType)
	{
		for (int i = 0; i < neurons; i++)
		{
			error[i] = (target[i] - output[i]) *
				derivative<T>(funcType, weighedSum<T>(input, matrix->getColWeights(i), prevNum) + arr[i]->getBias());
		}

	}

	void setError(T* errors, Matrix<T>* thisMatrix, Matrix<T>* prevMatrix, functionType funcType)
	{
		for (int i = 0; i < neurons; i++)
		{
			error[i] = (weighedSum(errors, thisMatrix->getStrWeights(i), arr[i]->getNextNum())) *
				(derivative<T>(funcType, weighedSum<T>(input, prevMatrix->getColWeights(i), prevNum) + arr[i]->getBias()));
		}
	}

	void setNeurons(int _neurons) { neurons = _neurons; }

	void add()
	{
		Neuron<T>* neuron = new Neuron<T>(prevNum, nextNum, 0);
		arr.add(*neuron);
		neurons++;
	}

	void del(int index)
	{
		arr.del(index);
		neurons--;
	}

	T* getInput() { return input; }

	T* getOutput() { return output; }

	T* getError() { return error; }

	T getError(int index) { return error[index]; }

	expArray<Neuron<T>> getNeurons() { return arr; }

	int getNextNum() { return nextNum; }

	int getPrevNum() { return prevNum; }

	int getNeuronsNum() { return neurons; }
};