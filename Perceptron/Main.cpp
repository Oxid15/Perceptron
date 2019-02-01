#include<iostream>
#include"Evolution.h"

void main()
{
	NeuralNet<float> net("testConfig.txt");
	net.addLayer(4, 1);
	net.fit("testData.csv", "testResults.csv", 4, 400);
	float e = net.validate("testData.csv", "testResults.csv", 4);
}