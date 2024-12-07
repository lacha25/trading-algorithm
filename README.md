trading algorithm based on momentum Trading and Algorithmic pattern recognition and bunch of over stuff. 

the buying decision is made over the value of the 5 previous signals (between -1 and 1) ###can normalize those value doesn't matter 

special formula is computed to compute the signal given all the algorithm, which should be easily updated, in case of 
removal or addition of some criterias. 

each algorithm should return a signal as well, that also fit the previous criterias(modulable)

f1:algorithm return data
f2:data is anaylsed and converted in a intermediate signal (NEED TO BE BETWEEN -1 and 1)
f3:global signal is computed
main: store the global signal
f4: algorithm to compute the decision
main: display the decision