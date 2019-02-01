#include<iostream>
#include"Evolution.h"

void main()
{
	NeuralNet<float> net("testConfig.txt");
	net.addLayer(2, 1);
	net.addNeuron(2, 1);

	//float e = net.validate("testData.csv", "testResults.csv", 4);
	net.fileOutput("currentConfig.txt");
}