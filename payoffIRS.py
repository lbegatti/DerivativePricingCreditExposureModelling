import numpy as np
from parameterDefinition import *

def payoff(row, column, cashflowNb, risk_free_rate, cpn_index, T):
    # cashflow present value
    cf_pv = 0.0
    cf_pv_ftp = 0.0
    if row == 0 and column == 0:
        print("The payoff function for both linear IRS and non-linear IRS has been called")
    if row in range(int(dcf * 250), timeStep + 1, int(dcf * 250)):
        cashflowNb = cashflowNb + 1
        disRate = risk_free_rate[row][column]
        for j in range(cashflowNb, T + 1):
            t = (j * (250 * dcf) - row) / float(250)
            df = np.exp(-disRate * t)
            df_ftp = np.exp(-(disRate + ftb_spread) * t)
            cf_raw = (cpn_index[row][column] + swapRateSpread - fixedLegCpn) * dcf * notional
            cf_pv = cf_pv + cf_raw * df
            cf_pv_ftp = cf_pv_ftp + cf_raw * df_ftp

        return cf_pv, cashflowNb, cf_pv_ftp
