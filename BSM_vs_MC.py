import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
#S0 is the stock price now
#K is the strike price
#T is the time of expiry in years
#r is the risk free rate
#s is sigma the volatility

#Computing analytical prices for our derivatives:

def derivatives_an(S0, k, T, r, s):
    d1 = (np.log(S0 / k) + (r + 0.5 * s ** 2) * T) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)

    call_an = S0 * st.norm.cdf(d1) - np.exp(-r * T) * k * st.norm.cdf(d2)

    put_an = - S0 * st.norm.cdf(- d1) + np.exp(-r * T) * k * st.norm.cdf(- d2)

    # For the binary put and call their price is just the disounted probability:

    bin_call_an = np.exp(-r * T) * st.norm.cdf(d1)
    bin_put_an = np.exp(-r * T) * st.norm.cdf(- d1)

    straddle_an = call_an + put_an

    RES = pd.DataFrame([call_an, put_an, straddle_an, bin_call_an, bin_put_an]).transpose()
    RES = RES.rename(columns = {0:'EU Call',1: 'EU Put', 2:'Straddle', 3:'Binary Call', 4:'Binary Put'})

    delta_call = st.norm.cdf(d1)
    delta_put = st.norm.cdf(d1) - 1
    return RES, delta_call, delta_put




#Note that all the above are in years.
#Consider 252 days in a year






#But in our function let's work in days:




def MC_price(S0, k, T, r , s, Np):

    #Inititalization:

    T_days = int(T*252)
    dt = 1
    r_days = r/252
    s_days = s/np.sqrt(252)
    S = np.zeros((T_days, Np))
    S[0, :] = S0


    #Simulation:
    for i in range(Np):
        z = np.random.standard_normal(T_days - 1)  #Random standard normal noise
        w = z*np.sqrt(dt)
        ret = np.cumsum((r_days - 0.5*s_days**2)*dt + w*s_days) #We define the daily returns
        S[1:, i] = S0*np.exp(ret)


    #We now have our matrix of simulated prices


    #Let's now price our derivatives:

    ST = S[-1, :]

    eu_call_temp = []
    eu_put_temp = []
    straddle_temp = []
    binary_call_temp = []
    binary_put_temp = []
    for i in range(len(ST)):
        eu_call_temp.append(max(ST[i] - k, 0)*np.exp(-r_days*T_days))
        eu_put_temp.append(max(k - ST[i], 0)*np.exp(-r_days*T_days))
        straddle_temp.append(eu_call_temp[i] + eu_put_temp[i])
        binary_call_temp.append((1 if eu_call_temp[i] > 0 else 0)*np.exp(-r_days*T_days)) #Assuming that the binary payoff would be 1 and 0
        binary_put_temp.append((1 if eu_put_temp[i] > 0 else 0)*np.exp(-r_days*T_days))   #Assuming that the binary payoff would be 1 and 0

    eu_call = np.mean(eu_call_temp)
    eu_call_se = np.std(eu_call_temp)
    eu_put = np.mean(eu_put_temp)
    eu_put_se = np.std(eu_put_temp)
    straddle = np.mean(straddle_temp)
    straddle_se = np.std(straddle_temp)
    binary_call = np.mean(binary_call_temp)
    binary_call_se = np.std(binary_call_temp)
    binary_put = np.mean(binary_put_temp)
    binary_put_se = np.std(binary_put_temp)

    a = pd.DataFrame({'Prices' : [eu_call, eu_put, straddle, binary_call, binary_put], 'Standard Errors' : [eu_call_se, eu_put_se, straddle_se, binary_call_se, binary_put_se]}, index=['EU Call', 'EU Put', 'Straddle', 'Binary Call', 'Binary Put'] )

    return a










MC_price(100, 102, 0.25, 0.1, 0.3, 100000)

derivatives_an(100, 102, 0.25, 0.1, 0.3)

#b


paths = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

prices = []
se = []

for ele in paths:
    df = MC_price(100, 102, 0.25, 0.1, 0.3, 2**ele)
    prices.append(df['Prices'][0])
    se.append(df['Standard Errors'][0])

#Plot1

plt.figure(figsize = (10,6))
plt.plot(paths,prices)
plt.grid()

plt.xlabel('Number of simulated paths Np as the number of powers of 2')
plt.ylabel('Price of vanilla call')
plt.title('Evolution of the prices of a vanilla call as a function of the number of simluated paths Np')
plt.show()

##Plot 2

plt.figure(figsize = (10,6))
plt.plot(paths,se)
plt.grid()

plt.xlabel('Number of simulated paths Np as the number of powers of 2')
plt.ylabel('Standard error of vanilla call')
plt.title('Evolution of the prices of a vanilla call as a function of the number of simluated paths Np')
plt.show()


#d

#Different random paths
delta_call = MC_price(101, 102, 0.25, 0.1, 0.3, 100000).iloc[0, 0] - MC_price(100, 102, 0.25, 0.1, 0.3, 100000).iloc[0, 0]
delta_put = MC_price(101, 102, 0.25, 0.1, 0.3, 100000).iloc[1, 0] - MC_price(100, 102, 0.25, 0.1, 0.3, 100000).iloc[1, 0]
#Same random paths:


def MC_delta(S0, k, T, r , s, Np):

    #Inititalization:

    T_days = int(T*252)
    dt = 1
    r_days = r/252
    s_days = s/np.sqrt(252)
    S = np.zeros((T_days, Np))
    S[0, :] = S0
    S_prime = np.zeros((T_days, Np))
    S0_prime = S0 + 1


    #Simulation:
    for i in range(Np):
        z = np.random.standard_normal(T_days - 1)  #Random standard normal noise
        w = np.cumsum(z)*np.sqrt(dt)
        ret = (r_days - 0.5*s_days**2)*dt + w*s_days #We define the daily returns
        S[1:, i] = S0*np.exp(ret)
        S_prime[1:, i] = S0_prime*np.exp(ret)

    #We now have our matrix of simulated prices


    #Let's now price our derivatives:

    ST = S[-1, :]
    ST_prime = S_prime[-1, :]
    eu_call_temp = []
    eu_put_temp = []
    eu_call_temp_prime = []
    eu_put_temp_prime = []
    for i in range(len(ST)):
        eu_call_temp.append(max(ST[i] - k, 0)*np.exp(-r_days*T_days))
        eu_put_temp.append(max(k - ST[i], 0)*np.exp(-r_days*T_days))
        eu_call_temp_prime.append(max(ST_prime[i] - k, 0)*np.exp(-r_days*T_days))
        eu_put_temp_prime.append(max(k - ST_prime[i], 0)*np.exp(-r_days*T_days))

    delta_call_prime = np.mean(eu_call_temp_prime) - np.mean(eu_call_temp)
    delta_put_prime = np.mean(eu_put_temp_prime) - np.mean(eu_put_temp)

    return delta_call_prime, delta_put_prime



delta_call_prime, delta_put_prime = MC_delta(100, 102, 0.25, 0.1, 0.3, 100000)
