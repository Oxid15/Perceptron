#include<fstream>
#include<string>

class FileName
{
	std::string name;
	std::string extension;
	int nameLen;
	int extLen;

public:

	FileName( std::string fileName)
	{
		char c = NULL;
		int i = 0;
		while (c != '.')
		{
			c = fileName[i];
			if (c != '.')
				name += fileName[i];
			i++;
		}
		nameLen = i - 1;

		int j = 0;
		while (c != '\0')
		{
			extension += fileName[i];
			c = fileName[i];
			i++;
			j++;
		}
		extLen = j - 1;
	}

	std::string getName() { return name; }

	std::string getExt() { return extension; }

	std::string getFullStr() { return name + '.' + extension; }

	void setName(std::string _name, int _length) 
	{ 
		name = _name;
		nameLen = _length;
	}

	void setExt(std::string _extension, int _length) 
	{ 
		extension = _extension; 
		extLen = _length;
	}
};

template<typename T>
void setrandomWeights(std::string inFileName, std::string outFileName, std::default_random_engine engine, int seed, T maxWeight, T minWeight = 0)
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
T** readCsv(std::fstream& file, int numOfStr, int numOfCol)
{
	T** output = new T*[numOfStr];
	for (int i = 0; i < numOfStr; i++)
		output[i] = new T;

	for (int i = 0; i < numOfStr; i++)
	{
		for (int j = 0; j < numOfCol; j++)
		{
			file >> output[i][j];
			file.get();
		}
	}

	return output;
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

template<typename T>
void normCsv(std::fstream& inFile, std::fstream& outFile, int strLen, int fileSize)
{
	for (int i = 0; i < fileSize; i++)
	{
		T* str = new T[strLen];
		str = readStrCsv<T>(inFile, strLen);
		str = normalizeVect<T>(str, strLen);

		writeStrCsv<T>(outFile, str, strLen);

		delete str;
	}
}