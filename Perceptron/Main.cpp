#include<iostream>
#include"NeuralNet.cpp"


int main()
{
	NeuralNet<double> net("testConfig.nn");
	double input[] = {1., 1.};
	double* output = new double;

	output = net.process(input);
	std::cout << output[0] << std::endl;
	std::system("pause");
}