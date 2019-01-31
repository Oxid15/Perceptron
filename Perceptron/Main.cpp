#include<iostream>
#include"Evolution.h"

void main()
{
	NeuralNet<float> net("testConfig.txt");
	net.fileOutput("currentConfig.txt");
	net.addLayer(4);
	net.fileOutput("currentConfig.txt");
}