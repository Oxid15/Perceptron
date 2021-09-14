#include<iostream>
#include"NeuralNet.cpp"


int main()
{
	NeuralNet<double> net("testConfig.nn");

	net.fit("data.csv", 4, "results.csv", 4, "data.csv", 4, "result.csv", 4, 500, 0.1);

	std::cout << "Test accuracy is: " << net.getEff() << std::endl;

	double inputs[4][2] = { {0., 0.}, {1., 0.}, {0., 1.}, {1., 1.} };
	double* output = new double;

	for (int i = 0; i < 4; i++)
	{
		net.process(inputs[i], output);
		std::cout << "Output when [" << inputs[i][0] << "," << inputs[i][1] << "]: " << *output << std::endl;
	}

	std::system("pause");
}