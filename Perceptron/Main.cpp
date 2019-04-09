#include<iostream>
#include"Evolution.cpp"

int main()
{
	//Population<float> p(2, 3, 5, -5, 10, 5);
	//p.mutation();
	//FileName name("net.txt");
	//p.fileOutput(name);
	//p.evaluate("testData.csv", "testResults.csv", "testData.csv", "testResults.csv", 4, 4, 400);

	//NeuralNet<> net("testConfig.nn");
	//net.fit("testData.csv", "testResults.csv", "testData.csv", "testResults.csv", 4, 4, 5000);

	//double arr[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	//elemPow(arr, 10, 2);
	//double d = euclidNorm<double>(arr, 10);
	//return 0;

	//expArray<int> arr;
	//int a = 1;
	//int b = 2;
	//int c = 3;
	//int d = 4;
	//arr.add(a);
	//arr.add(b);
	//arr.add(c);
	//arr.add(d);
	//for (int i = 0; i < 4; i++)
	//{
	//	std::cout << *arr[i] << "\n";
	//}

	//expArray<int*> arr;
	//int* a = new int[3];
	//a[0] = 1; a[1] = 2; a[2] = 3;
	//int* b = new int[3];
	//b[0] = 4; b[1] = 5; b[2] = 6;
	//int* c = new int[3];
	//c[0] = 7; c[1] = 8; c[2] = 9;
	//arr.add(a);
	//arr.add(b);
	//arr.add(c);
	//int* aa = *arr[0];
	//int* bb = *arr[1];
	//int* cc = *arr[2];
	//std::cout << aa[0] << " " << aa[1] << " " << aa[2] << "\n";
	//std::cout << bb[0] << " " << bb[1] << " " << bb[2] << "\n";
	//std::cout << cc[0] << " " << cc[1] << " " << cc[2] << "\n";

	std::fstream file("test.csv");
	double** arr = readCsv<double>(file, 78, 1);
	double* a = new double[78];
	for (int i = 0; i < 78; i++)
		a[i] = arr[i][0];

	int* freq = new int[10];
	computeFrequencies<double>(freq, a, 78, 10);
	int sum = 0;
	for (int i = 0; i < 10; i++)
	{
		sum += freq[i];
		std::cout << freq[i] << "\n";
	}
	std::cout << sum;
	std::cout << "\n";

	double* Cmdist = new double[10];
	computeCmltvDistFunc(Cmdist, a, 78, 10);
	for (int i = 0; i < 10; i++)
	{
		std::cout << Cmdist[i] << "\n";
	}
	std::cout << "\n";

	double sum2 = 0;
	double* denseFunc = new double[10];
	computeDenseFunc(denseFunc, a, 78, 10);
	for (int i = 0; i < 10; i++)
	{
		sum2 += denseFunc[i];
		std::cout << denseFunc[i] << "\n";
	}
	std::cout << sum;
	std::cout << "\n";
}