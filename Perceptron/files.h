#include<fstream>
#include<string>

template<typename T>
void setrandomWeights(std::string inFileName, std::string outFileName, int seed, T maxWeight, T minWeight = 0)
{
	std::ifstream inFile(inFileName);
	std::ofstream outFile(outFileName);
	static std::default_random_engine engine;
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
	outFile << type   << "\n";

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
				outFile << randomNumber<T>(seed, engine, maxWeight, minWeight) << " ";
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
				outFile << randomNumber<T>(seed, engine, maxWeight, minWeight) << " ";
				seed++;
			}
		}
		outFile << "\n";
	}
}

template<typename T>
T* readStrCsv(std::fstream& file, int length)			
{
	T* output = new T[length];
	for (int i = 0; i < length; i++)
	{
		file >> output[i];
		file.get();
	}
	return output;
}

template<typename T>
void writeStrCsv(std::fstream& file, T* str, int length)
{
	for (int i = 0; i < length; i++)
	{
		file << str[i];
		file << ';';
	}
	file << "\n";
}