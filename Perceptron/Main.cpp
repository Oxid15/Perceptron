#include<iostream>
#include"Evolution.h"

int main()
{
	//Population<float> p(2, 3, 5, -5, 10, 5);
	//p.mutation();
	//FileName name("net.txt");
	//p.fileOutput(name);
	////p.evaluate("testData.csv", "testResults.csv", "testData.csv", "testResults.csv", 4, 4, 400);

	NeuralNet<> net("testConfig.txt");
	net.fit("testData.csv", "testResults.csv", "testData.csv", "testResults.csv", 4, 4, 500);
	std::cout << net.getEff();
	system("pause");
	return 0;
}