#include<iostream>
#include<fstream>
#include<string>

template<typename T>
void setrandomWeights
(
	std::string inFileName, 
	std::string outFileName, 
	std::default_random_engine engine, 
	T maxWeight,
	int seed = 0,
	T minWeight = 0
)
{
	std::ifstream inFile(inFileName);
	std::ofstream outFile(outFileName);
	int layers;
	inFile >> layers;

	int type;
	inFile >> type;

	int* neurons = new int[layers];
	for (int i = 0; i < layers; i++)
		inFile >> neurons[i];

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
				outFile << unifRealRandNum<T>(seed, engine, maxWeight, minWeight) << " ";
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
				outFile << unifRealRandNum<T>(seed, engine, maxWeight, minWeight) << " ";
				seed++;
			}
		}
		outFile << "\n";
	}
	delete neurons;
}

template<typename T>
void readCsv(std::fstream& file,T** output, int numOfStr, int numOfCol)
{
	for (int i = 0; i < numOfStr; i++)
	{
		for (int j = 0; j < numOfCol; j++)
		{
			file >> output[i][j];
			file.get();
		}
	}
}

//too simple - can crash sometimes 
//and doesn't have an exception handling
template<typename T>
void readStrCsv(std::fstream& file,T* output, int strLen)
{
	for (int i = 0; i < strLen; i++)
	{
		file >> output[i];
		file.get();
	}
}

template<typename T>
void writeStrCsv(std::fstream& file, T* str, int strLen)
{
	for (int i = 0; i < strLen; i++)
	{
		file << str[i];
		file << ';';
	}
	file << "\n";
}

//normalizes numerical vectors in csv file by euclidean distance
template<typename T>
void normCsv(std::fstream& inFile, std::fstream& outFile, int strLen, int numOfStr)
{
	for (int i = 0; i < numOfStr; i++)
	{
		T* str = new T[strLen];
		str = readStrCsv<T>(inFile, strLen);
		str = normalizeVect<T>(str, strLen);
		writeStrCsv<T>(outFile, str, strLen);
		delete str;
	}
}