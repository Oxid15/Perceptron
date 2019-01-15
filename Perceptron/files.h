#include<fstream>
#include<string>

template<typename T>
void setrandomWeights(std::string inFileName, std::string outFileName, int seed, T range)
{
	std::ifstream inFile(inFileName);
	std::ofstream outFile(outFileName);
	int layers;
	inFile >> layers;

	int type;
	inFile >> type;

	int* neurons = new int[layers];
	for (int i = 0; i < layers; i++)
	{
		inFile >> neurons[i];
	}

	outFile << layers << "\n";

	outFile << type << "\n";

	for (int i = 0; i < layers; i++)
	{
		outFile << neurons[i];
		if (i != layers - 1)
			outFile << " ";
	}
	outFile << "\n";

	for (int i = 0; i < layers; i++)
	{
		for (int j = 0; j < neurons[i]; j++)
		{
			if (i)
				outFile << randomNumber<T>(seed, range) << " ";
			else
				outFile << "0 ";
			seed++;
		}
		outFile << "\n";
	}

	for (int k = 0; k < layers - 1; k++)
	{
		for (int i = 0; i < neurons[k]; i++)
		{
			for (int j = 0; j < neurons[k + 1]; j++)
			{
				outFile << randomNumber<T>(seed, range) << " ";
				seed++;
			}
		}
		outFile << "\n";
	}
}


template<typename T>
T* getStrFromCsv(std::fstream& file, int length)
{
	T* output = new T;
	for (int i = 0; i < length; i++)
	{
		file << output[i];
		file.get();
	}
	return output;
}

template<typename T>
T* getStrFromFile(std::ifstream& trainSet)
{
	int size = arrLayers[0]->getNeuronsNum();
	T* output = new T[size];
	for (int i = 0; i < size; i++)
	{
		output[i] = getValueFromCsv(trainSet);
	}
	return output;
}

template<typename T>
T getAccuracyFromFile(std::string fileName, int output_length, int size)
{
	std::ifstream file(fileName);
	T all = 0;
	T correct = 0;
	T eff;
	for (int p = 0; p < size; p++)
	{
		for (int k = 0; k < output_length; k++)
		{
			T* output = new T[output_length];
			T* target = new T[output_length];
			for (int i = 0; i < output_length; i++)
			{
				output[i] = getValueFromCsv(file);
			}
			for (int i = 0; i < output_length; i++)
			{
				output[i] = getValueFromCsv(file);
			}
			for (int i = 0; i < output_length; i++)
			{
				if ((output[i] > 0 && output[i] < 1) &&
					(target[i] >= 0 && target[i] <= 1))
				{
					if (output[i] >= 0.5)
						output[i] = 1;
					else
						output[i] = 0;
				}

				if (output[i] == target[i])
					correct++;
			}
			all++;
			delete output;
			delete target;
		}
	}
	return eff = correct / all;
}
