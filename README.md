# Polynomial-Back-Propagation-Algorithm

Overview: 

This algorithm samples the presecified full space of weight and bias values plotted against their corresponding losses within the context of the larger network. The algorithm then preforms a degree 4 polynomial regression to determine the localised gradient. The algorithm proved most effective when paired with a randomization of node optimization order. 

Results:

Using my own neural network implementation with only a single hidden perceptron the algorithm demonstrated impressive convergence on the XOR problem. These results are all the more inspiring given the stupidity of my perceptron relative to other models. Below is a graph of the loss plotted against the training epochs.

<img width="572" alt="Screen Shot 2023-05-10 at 11 02 03 AM" src="https://github.com/J-sandler/Polynomial-Back-Propagation-Algorithm/assets/108235294/d0c004d6-e6b4-40cf-9b9d-5a47c501bbe5">

The algorithm preformed 10000 epochs (roughly equating to 10s of train time) at which point its preformance on the xor problem was as follows: 

<img width="283" alt="Screen Shot 2023-05-10 at 11 38 12 AM" src="https://github.com/J-sandler/Polynomial-Back-Propagation-Algorithm/assets/108235294/a1742092-0773-4260-9e77-9a0e786da97c">

The network finds a rather unimpressive solution to the problem given the perceptrons inability to linearly separate. 

However, the network was obviously able to solve the problem extremely well when gifted this ability. By using a layer of 10 perceptrons instead of 1 the algorithm converged to its minimum stage within only 42 epochs, with a loss of 1.0196330739781823e-05. The convergence was comparitevly rapid as well:

<img width="560" alt="Screen Shot 2023-05-10 at 11 30 03 AM" src="https://github.com/J-sandler/Polynomial-Back-Propagation-Algorithm/assets/108235294/e835b950-686a-40a4-aede-cc2912240d48">

Here is the output for one of these stages:

<img width="281" alt="Screen Shot 2023-05-10 at 11 43 07 AM" src="https://github.com/J-sandler/Polynomial-Back-Propagation-Algorithm/assets/108235294/003f637b-c857-4cae-bd63-740a5259d161">
