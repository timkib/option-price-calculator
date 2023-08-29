### MAIN streamlit file
import streamlit as st
from calculations import BlackScholesMarket, HestonModel
import numpy as np

st.title('Option price calculator')
st.markdown('Welcome to the option price calculator. It offers you a variety of option types including european options and american.')

# default parameters
t = 0
S0 = 110
S_t = S0
r = 0.05
sigma = 0.3
T = 1
K = 100
M = 1000
sigma_tilde = 0.5
lambda_parameter = 2.5
m = 500
kappa = 0.5
gamma0 = np.square(0.2)

# create instance
option_pricing_BS = BlackScholesMarket(t, S0, S_t, r, sigma, T, K, M)
option_pricing_Heston = HestonModel(t, S0, S_t, r, sigma_tilde, T, K, M, lambda_parameter, m, kappa, gamma0)
# col1, col2, col3 = st.columns(3)
# with col1:
#     S_t = st.number_input('Current stock price', min_value=0.0001)
# with col2:
#     K = st.number_input('Strike price', min_value=0.001)

# with col3:
#     sigma = st.number_input('Standard Deviation (%)', min_value=0.0001)
#     sigma = sigma / 100


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



