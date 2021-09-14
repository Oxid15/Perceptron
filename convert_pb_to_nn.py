import os
from tensorflow.keras import models
import argparse


ROOT_DIR = os.path.abspath(__file__)
print(ROOT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str)
args = parser.parse_args()

model = models.load_model(args.p)
model.compile()
model.build()

layers_num = len(model.layers)

neurons = []
for layer in model.layers:
    neurons.append(len(layer.weights[0][:, 0]))
neurons.append(len(model.layers[-1].weights[1].numpy()))

with open(os.path.join(args.p, 'saved_model.nn'), 'w') as f:
    f.write(str(layers_num)+'\n')
    f.write('0\n')
    for neuron in neurons:
        f.write(str(neuron) + ' ')
    f.write('\n')

    for _ in range(neurons[0]):
        f.write('0.000000 ')
    f.write('\n')
    for n in range(1, len(neurons)):
        biases = model.layers[n - 1].weights[1].numpy().flatten()
        for i in range(neurons[n]):
            f.write(f'{biases[i]:.6f} ')
        f.write('\n')

    for n in range(1, len(neurons)):
        for i in range(neurons[n]):
            f.write(f'{model.layers[n - 1].weights[0].numpy().flatten()[i]:.6f} ')
    
    for n in range(1, len(neurons)):
        weights = model.layers[n - 1].weights[0].numpy().flatten()
        for i in range(len(weights)):
            f.write(f'{weights[i]:.6f} ')
        f.write('\n')

print(f'Done, file is: {os.path.join(args.p, "saved_model.nn")}')
