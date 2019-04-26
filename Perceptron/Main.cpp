//TODO change name of all "size" function parameters to more appropriate ones

#include<iostream>
#include"Evolution.cpp"
int main()
{
	std::fstream file("test.csv");
	double** arr = readCsv<double>(file, 78, 1);
	double* a = new double[78];
	for (int i = 0; i < 78; i++)
		a[i] = arr[i][0];
	
	std::cout << mean(a, 78) << "\n";
	std::cout << median(a, 78) << "\n";
	std::cout << variance(a, 78, 0) << "\n";
	std::cout << variance(a, 78, 1) << "\n";
}