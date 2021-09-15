# Perceptron
**Perceptron** — mathematical model of brain perception, cybernetic brain model also known as neural net in machine learning. It was invented in 1958 by Frank Rosenblatt and was implemented as electronic machine "Mark-1". "Mark-1" was the first neurocomputer in the world.

https://en.wikipedia.org/wiki/Perceptron

## This repository:
In this repo, multilayer perceptron (MLP) is implemented. The program allows to create and train neural networks with arbitrary number of layers and neurons. Python scripts allow to convert neural nets from Tensorflow .pb format to the custom format of this repository with the extension .nn. The inference results should remain the same.

## How to use the code
There is several usage scenarios for this repository. It can be used for lightweight C++ Tensorflow
inference and also independently.

### You have tensorflow model
In this case you can use `convert_pb_to_nn.py` to convert TF model to .nn format. This feature is tested with tensorflow 2.4.1.
Then you can use `saved_model.nn` as a config for neural net in Perceptron project.  
You can find sample of usage in `Main.cpp`.  
```cpp
#include<iostream>
#include"NeuralNet.cpp"


int main()
{
    // Create neural net with your config
	NeuralNet<double> net("saved_model.nn");

    // Initialize input of the shape that was described in .nn file 
	double inputs[2] = {0., 0.};

    // Initialize output
	double* output = new double;

    // Inference the model
    net.process(inputs, output);

    std::cout << "Output with [" << inputs[0] << "," << inputs[1] << "]: " << *output << std::endl;
}
```

### You don't have tensorflow model
You can use this repository independently. Create and train models with simple gradient descent algorithm that is implemented.

```cpp
#include<iostream>
#include"NeuralNet.cpp"


int main()
{
    // Create model with the sample config file or create your own
    // automatic creation of this files is not implemented, so the creation process can be sophisticated
    // the structure of the .nn files is given below
	NeuralNet<double> net("testConfig.nn");

    // Train the model with train data from data.csv and using targets from results.csv
    // second pair of files is the testing data
    // numbers passed after file names are the sizes of the files
	net.fit("data.csv", 4, "results.csv", 4, "data.csv", 4, "result.csv", 4, /*epochs*/500);

	std::cout << "Test accuracy is: " << net.getEff() << std::endl;

    // Initialize input of the shape that was described in .nn file
	double inputs[4][2] = { {0., 0.}, {1., 0.}, {0., 1.}, {1., 1.} };

    // Initialize output
	double* output = new double;

	for (int i = 0; i < 4; i++)
	{
        // Inference the model
		net.process(inputs[i], output);
		std::cout << "Output when [" << inputs[i][0] << "," << inputs[i][1] << "]: " << *output << std::endl;
	}
}
```

### Description of custom format
`.nn` is the simple text format to store the weights and architecture of simple dense neural nets.  
Let's see the example of the file:
```
3                           # the number of layers
0                           # this encodes activation functions for the corresponding enum: for example 0 is sigmoid
2 3 1                       # this is the number of neurons in every layes
0.000000 0.000000           # this and two lines below is the biases of the neurons of every layer - input layer must have zero biases
-2.750024 -0.543923 -1.436713 
-3.001153 
1.675041 3.106899 4.541039 2.750963 5.291639 6.520514 # this is weights between first and second layer - there are 2*3=6 weights
-3.596334 1.552888 3.792343 # this is weights between second and third layer - there are 3*1=3 weights
```
