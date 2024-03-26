Perceptron (Logic Gates)

Given two inputs: 
- an `AND` gate (`A AND B = X`) will output a 1 only if both inputs are a 1;
- an `OR` gate (`A OR B = X`) will output a 1 if any input is a 1;
- an `XOR` gate (`A XOR B = X`) will outputs a 1 if only one input is a 1.

Goals:
- Use perceptrons to model the fundamental building blocks of computers — logic gates.
- See how an `AND` gate can be thought of as linearly separable data, and train a perceptron to perform `AND`.
- See why an `XOR` gate isn't linearly separable, and show how a perceptron fails to learn `XOR`.