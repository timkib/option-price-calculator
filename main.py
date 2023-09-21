### MAIN streamlit file
import streamlit as st
from calculations import BlackScholesMarket, HestonModel
import numpy as np

st.title('Option price calculator')
st.markdown('Welcome to the option price calculator. It offers you a variety of option types including european options and american.')

# default parameters
t = 0
# S0 = 110
# S_t = S0
# r = 0.05
# sigma = 0.3
T = 1
K = 100
M = 1000
sigma_tilde = 0.5
lambda_parameter = 2.5
m = 500
kappa = 0.5
gamma0 = np.square(0.2)


col1, col2, col3, col4 = st.columns(4)
with col1:
    S0 = st.number_input('Current stock price', min_value=0.0001)
    S_t = S0
with col2:
    K = st.number_input('Strike price', min_value=0.001)

with col3:
    r = st.number_input('Interest rate (%)', min_value=0.001)

with col4:
    sigma = st.number_input('Standard Deviation (%)', min_value=0.0001)
    sigma = sigma / 100

# create instance
option_pricing_BS = BlackScholesMarket(t, S0, S_t, r, sigma, T, K, M)
option_pricing_Heston = HestonModel(t, S0, S_t, r, sigma_tilde, T, K, M, lambda_parameter, m, kappa, gamma0)


col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    option_type = st.radio(
        "What type of option?",
        ('Vanilla option', 'Barrier option'))

with col2:
    excercise_type = st.radio(
        "What type of exercise?",
        ('European style', 'American style'))
    
with col3:
    model_type = st.radio(
        "What model type?",
        ('Black-Scholes market', 'Heston model'))

with col4:
    solving_method = st.radio(
        "Solving method?",
        ('Explicit formula', 'Monte-Carlo', 'PDE-method', 'Integral transformation'))

# check if barrier option was selected
if option_type == "Barrier option":
    barrier_H = st.number_input('Enter barrier', min_value=0.1)
    option_pricing_BS.H = barrier_H

try:
    if st.button('Calculate price'):
        if excercise_type == "European style" and model_type == "Black-Scholes market":
            if option_type == "Vanilla option":
                if solving_method == "Explicit formula":
                    V_0_call = option_pricing_BS.BlackScholes_EuCall() # call option
                    V_0_put = option_pricing_BS.BlackScholes_EuPut() # put option
                elif solving_method == "Monte-Carlo":
                    V_0_call = option_pricing_BS.Eu_Option_BS_MC("Call") # call option
                    V_0_put = option_pricing_BS.Eu_Option_BS_MC("Put") # put option
                elif solving_method == "PDE-method":
                    stock_matrix, option_price_matrix  = option_pricing_BS.Eu_Option_BS_PDE() # call option
                    idx = np.absolute(stock_matrix-S_t).argmin()
                    V_0_call = option_price_matrix[idx]
                    V_0_put = 0 # put option  #TODO
                elif solving_method == "Integral transformation":
                    V_0_call  = option_pricing_BS.Eu_Option_BS_LaPlace("Call") # call option
                    V_0_put = option_pricing_BS.Eu_Option_BS_LaPlace("Put") # put option 

            elif option_type == "Barrier option":
                if solving_method == "Explicit formula":
                    V_0_call = option_pricing_BS.BlackScholes_Eu_Barrier_Call() # call option
                    V_0_put = option_pricing_BS.BlackScholes_Eu_Barrier_Put() # put option
                elif solving_method == "Monte-Carlo":
                    V_0_call = option_pricing_BS.Eu_Option_BS_MC_Barrier("Call") # call option
                    V_0_put = option_pricing_BS.Eu_Option_BS_MC_Barrier("Put") # put option

        elif excercise_type == "European style" and model_type == "Heston model":
            if option_type == "Vanilla option":
                if solving_method == "Explicit formula":
                    st.warning('There is no closed form solution. Try Monte-Carlo method.')
                    V_0_call = 0 # call option
                    V_0_put = 0 # put option
                elif solving_method == "Monte-Carlo":
                    V_0_call = option_pricing_Heston.Heston_EuCall_MC_Euler("Call") # call option
                    V_0_put = option_pricing_Heston.Heston_EuCall_MC_Euler("Put") # put option
                elif solving_method == "Integral transformation":
                    V_0_call = option_pricing_Heston.Heston_EuCall_MC_LaPlace("Call") # call option
                    V_0_put = option_pricing_Heston.Heston_EuCall_MC_LaPlace("Put") # put option
        
        elif excercise_type == "American style" and model_type == "Black-Scholes market":
            if solving_method == "PDE-method":
                stock_matrix_call, option_price_matrix_call  = option_pricing_BS.Am_Option_BS_PDE("Call") # call option
                idx = np.absolute(stock_matrix_call-S_t).argmin()
                V_0_call = option_price_matrix_call[idx]

                stock_matrix_put, option_price_matrix_put  = option_pricing_BS.Am_Option_BS_PDE("Put") # call option
                idx = np.absolute(stock_matrix_put-S_t).argmin()
                V_0_put = option_price_matrix_put[idx]
            
            elif solving_method == "Monte-Carlo":
                V_0_call = option_pricing_BS.Am_Option_BS_LS("Call") # call option
                V_0_put = option_pricing_BS.Am_Option_BS_LS("Put") # put option

        elif excercise_type == "American style" and model_type == "Heston model":
            if solving_method == "Monte-Carlo":
                V_0_call = option_pricing_Heston.Am_Option_Heston_LS("Call") # call option
                V_0_put = option_pricing_Heston.Am_Option_Heston_LS("Put")  # put option 
            
        st.text(f"Call price: {np.round(V_0_call, 2)} | Put price: {np.round(V_0_put, 2)}")
except:
   st.text("This combination is not possible. Try another method.") 


st.header('How does it work?')


st.write('Vanilla Option: The classical option with the following payout:')
st.latex(r'''
    {(S - K, 0)^+ } 
    ''')
st.write('Where S denotes the stock price and K the strike price, the + indicates the positive part.')
st.write('Barrier Option: the down-and-out option means, that if the stock price is below a certain value, it loses it reduces its vale to 0 for the rest of the period.')
st.write('Monte-Carlo: ')
st.latex(r'''
    {V^{} = \frac{1}{N} \sum_{n=1}^{N} f(X_n) } 
    ''')
st.write("The idea is to approximate integrals or expectations using the empirical counterpart. It makes sense since the MC estimator is unbiased and converges in probability to the true value. For complex payouts or path dependent option it is a fast method. However, many realizations are necessary i.e. N should be large to accomplish decent results. For american options it uses the Longstaff-Schwartz algorithm.")
st.write('PDE-method: ')
st.write('The idea is to use the black-scholes / heston pde and then using a grid where the derivatives are discretized using finite differences. The following calcualtor uses the crank nicolson scheme. For american options it uses the Brennan–Schwartz algoritm. PDE-methods are very accurate and stable, however with a finer time steps and more assets it takes exponentially more time.')
st.write('Integral transformation: ')
st.write('It transforms the payout using the LaPlace transformation and includes the characteristic function of the stock price. Then it inverts the term using Mellin transform. The integral is solved numerically. Integral transformation are efficient, but cannot handle path dependent options.')

st.write('American options: ')
st.write('By contrast, american options can be exercised at any point before maturity. The pricing is based on optimal stopping i.e.')
st.latex(r'''
    {V(t) = S_0(t) \sup_{\tau} \mathbb{E}\left[ \frac{X(\tau)}{S_0(\tau)} \bigg| \mathcal{F}_t \right]
 } 
    ''')
st.write('S_0 denotes the prices the bond for discounting, tau is the optimal stopping time giving the natural filtration F_t.')
st.write('Heston model: ')
st.write('The main difference is that in the heston model, the volatility is not assumed to constant but replaced by a stochastic process. This is motivated by the fact that in real markets the volatility of an asset is not constant.')
st.write("All calculations are based on my OptionPricing package, for more information see my github [link](https://github.com/timkib/OptionPricing)")

