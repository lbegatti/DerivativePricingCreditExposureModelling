import numpy as np
import random


## random number generation
def randomNumber_1(n_iterations):
    """
        Definition
        ----------
        Python method to generate a random number. The formula is an approx of the Normal distribution.
            random.random() draws random number from a uniform distribution [0,1).


        Parameter
        ---------
        n: number of loops to make the result meaningful.


        Return
        ------
        random number generated

    """
    a = []
    for i in range(0, n_iterations):
        a.append(random.random())
    rn = np.sqrt(12.0 / float(n_iterations)) * (np.sum(a) - n_iterations / 2.0)
    return rn
