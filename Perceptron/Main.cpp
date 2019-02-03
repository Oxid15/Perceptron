#include<iostream>
#include"Evolution.h"

void main()
{
	NeuralNet<float> net("testConfig.txt");

	
	net.delLayer(2, 1);
	net.delLayer(1, 1);

	net.fit("testData.csv", "testResults.csv", 4, 500);
	net.getEff();

	net.fileOutput("currentConfig.txt");
}