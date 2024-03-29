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

	T* process(T* _input, Matrix<T>* matrix, functionType ftype)
	{
		input = _input;
		for (int i = 0; i < neurons; i++)
		{
			T* weights = new T[matrix->getHeight()];
			matrix->getColWeights(weights, i);
			output[i] = arr[i]->process(_input, weights, ftype);
			delete[] weights;
		}
		return output;
	}

	void setError(T* target, Matrix<T>* matrix, functionType ftype)
	{
		for (int i = 0; i < neurons; i++)
		{
			T* weights = new T[matrix->getHeight()];
			matrix->getColWeights(weights, i);
			error[i] = (target[i] - output[i]) *
				derivative<T>(ftype, weighedSum<T>(input, weights, prevNum) + arr[i]->getBias());
			delete[] weights;
		}
	}

	void setError(T* errors, Matrix<T>* thisMatrix, Matrix<T>* prevMatrix, functionType ftype)
	{
		for (int i = 0; i < neurons; i++)
		{
			T* strWeights = new T[thisMatrix->getLength()];
			thisMatrix->getStrWeights(strWeights, i);

			T* colWeights = new T[prevMatrix->getHeight()];
			prevMatrix->getColWeights(colWeights, i);

			error[i] = (weighedSum(errors, strWeights, arr[i]->getNextNum())) *
				(derivative<T>(ftype, weighedSum<T>(input, colWeights, prevNum) + arr[i]->getBias()));
			delete[] strWeights;
			delete[] colWeights;
		}
	}

	void setNeurons(int _neurons) { neurons = _neurons; }

	T* getInput() { return input; }

	T* getOutput() { return output; }

	T* getError() { return error; }

	T getError(int index) { return error[index]; }

	expArray<Neuron<T>> getNeurons() { return arr; }

	int getNextNum() { return nextNum; }

	int getPrevNum() { return prevNum; }

	int getNeuronsNum() { return neurons; }
};