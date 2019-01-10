#include"Evolution.cpp"

int main()
{
	NeuralNet<float>* n = init<float>(3, 3);
	n[1].fileOutput("currentConfig.txt");
	return 0;
}