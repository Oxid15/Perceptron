#include<iostream>
#include"Evolution.h"

void main()
{
	NeuralNet<float> net("testConfig.txt");
	net.addNeuron(1, 1);
	net.addNeuron(1, 1);
	net.addLayer(3, 1);
	net.addNeuron(2, 1);

	net.fit("testData.csv", "testResults.csv", 4, 5000);
	net.getEff();
	net.fileOutput("currentConfig.txt");
}