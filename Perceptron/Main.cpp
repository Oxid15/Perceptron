#include<iostream>
#include"Evolution.cpp"

void main()
{
	NeuralNet<float> n("testConfig.txt");
	n.fileOutput("currentConfig.txt");
}