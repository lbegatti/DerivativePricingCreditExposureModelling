import numpy as np
from randomNumberGeneration import randomNumber_1

# for random number generations
n_iterations = 12


class riskFactorModelling:
    """
        Definition
        ----------
        Python class for Risk factor modelling. The model is either the Vasicek or GMB.

        It accomodates the correlation parameters for more than 1 risk factor and the volatility
        is not constant but Heston-based.

        Furthermore, the model has the feature to allow the same or a different Heston vol for the risk factor.

    """

    def __init__(self, vol, dt):
        """
            Definition
            ----------
            Initialization of the riskFactorModelling class.


            Parameter
            ----------
            vol: spot volatility, i.e. observable in the market -> (so variance is: vol**2)
            dt: time increments for the stochastic differential equations.


            Return
            -------
            None

        """
        self.vol = vol
        self.dt = dt
        self.dB = randomNumber_1(n_iterations) * np.sqrt(self.dt)

    def hestonVol(self):
        """
            Definition
            ----------
            Python method for the Heston Stochastic Volatility Function.


            Parameter
            ---------
            None


            Return
            ------
            Variance (as the model simulates the variance, not the vol directly)

        """
        # mean reversion parameter
        theta = 0.5

        # long term volatility
        w = 0.16

        # vol of vol
        etha = 0.2

        if 2 * theta * w <= etha ** 2:
            raise Exception("Feller condition breached for Heston Model")

        variance_tminus1 = self.vol ** 2

        # variance_t = variance_tminus1 + time increment (dV_t) (Heston)
        variance_t = \
            variance_tminus1 + theta * (w - variance_tminus1) * self.dt + etha * np.sqrt(variance_tminus1) * self.dB

        return variance_t

    def geometricBrownian(self, s_tminus1, resetVol=False, rateCorrBM=9.99):
        """
            Definition
            ----------
            Python method for the Geometric Brownian motion.


            Parameter
            ---------
            s_tminus1: underlying price at T-1

            resetVol: boolean, if False the volatility is the same, if True the method SetVol() is called to generate
                a different volatility using hestonVol().

            rateCorrBM: default value correlation to calculate a CORRELATED Brownian Motion, i.e.
                between two risk factors.


            Return
            ------
            s_t -> underlying (future) price at T {stochastic process}

        """
        # drift parameter
        mu = 0.05

        #
        self.setCorrW(rateCorrBM)
        self.setVol(resetVol)

        # price_t = price_tminus1 + time increment (S_t) (GBM)
        s_t = s_tminus1 * np.exp((mu - self.vol ** 2 / 2) * self.dt + self.vol * self.dB)

        return s_t

    def vasicek(self, r_tminus1, resetVol=False, rateCorrBM=9.99):
        """
            Definition
            ----------
            Python method to implement the Vasicek (mean reverting model ) model which describes the evolution of interest rates. It is
                1-factor short rate model as it describes the interest rate movements as driven by only one source
                of market risk.


            Parameter
            ---------
            r_tminus1: underlying rate at T-1

            resetVol: boolean, if False the volatility is the same, if True the method SetVol() is called to generate
                a different volatility using hestonVol().

            rateCorrBM: default value correlation to calculate a CORRELATED Brownian Motion, i.e.
                between two risk factors.


            Return
            ------
            r_t -> underlying (future) rate at T {stochastic process}

        """
        # mean reversion speed
        a = 0.5

        # long term yield curve target
        b = 0.04

        self.setCorrW(rateCorrBM)
        self.setVol(resetVol)

        # rate_t = rate_tminus1 + time increment (R_t) (Vasicek)
        r_t = r_tminus1 + a * (b - r_tminus1) * self.dt + self.vol * self.dB

        return r_t

    def setCorrW(self, rateCorrBM):
        """
            Definition
            ----------
            Python method to set the correlation to calculate the CORRELATED Brownian Motion. The default value is 9.99.


            Parameter
            ---------
            rateCorrBM: default value correlation to calculate a CORRELATED Brownian Motion, i.e.
                between two risk factors.


            Return
            ------
            None.

        """
        if rateCorrBM == 9.99:
            self.dB = self.dB
        else:
            dZ = randomNumber_1(n_iterations) * np.sqrt(self.dt)
            self.dB = rateCorrBM * self.dB + np.sqrt(1 - rateCorrBM ** 2) * dZ

    def setVol(self, resetVol):
        """
            Definition
            ----------
            Python method to generate a new vol with the hestonVol().


            Parameter
            ---------
            resetVol: boolean, if False the volatility is the same, if True the method SetVol() is called to generate
                a different volatility using hestonVol().


            Return
            ------
            None.

        """
        if resetVol:
            self.vol = np.sqrt(self.hestonVol())
