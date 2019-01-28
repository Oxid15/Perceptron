#include<iostream>
#include"Evolution.h"

void main()
{
	NeuralNet<float> net("testConfig.txt");
	net.fit("testData.csv", "testResults.csv", 4, 400);
	net.fileOutput("currentConfig.txt");
	std::cout << net.validate("testData.csv", "testResults.csv", 4);
}