//TODO change name of all "size" function parameters to more appropriate ones

#include<iostream>
#include"Evolution.cpp"
int main()
{
	std::fstream file("test2.csv");
	double** arr = readCsv<double>(file, 67, 1);
	double* a = new double[67];
	for (int i = 0; i < 67; i++)
		a[i] = arr[i][0];
	
	std::cout << "              mean             = " << mean(a, 67) << "\n";
	std::cout << "              median           = " << median(a, 67) << "\n";
	std::cout << "[shifted]     variance         = " << variance(a, 67, true) << "\n";
	std::cout << "[not_shifted] variance         = " << variance(a, 67, false) << "\n";
	std::cout << "[shifted]     SD               = " << SD(a, 67, true) << "\n";
	std::cout << "[not_shifted] SD               = " << SD(a, 67, false) << "\n";
	std::cout << "              measure of scale = " << MeasOfScale(a, 67) << "\n";
	std::cout << "\n";
	std::cout << "[shifted][raw]             1-th moment = " << rawKthMoment(a, 1, 67) << "\n";
	std::cout << "[shifted][raw]             2-th moment = " << rawKthMoment(a, 2, 67) << "\n";
	std::cout << "[shifted][raw]             3-th moment = " << rawKthMoment(a, 3, 67) << "\n";
	std::cout << "[shifted][raw]             4-th moment = " << rawKthMoment(a, 4, 67) << "\n";
	std::cout << "\n";
	std::cout << "[shifted][central]         1-th moment = " << centralKthMoment(a, 1, 67) << "\n";
	std::cout << "[shifted][central]         2-th moment = " << centralKthMoment(a, 2, 67) << "\n";
	std::cout << "[shifted][central]         3-th moment = " << centralKthMoment(a, 3, 67) << "\n";
	std::cout << "[shifted][central]         4-th moment = " << centralKthMoment(a, 4, 67) << "\n";
}