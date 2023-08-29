### methods to calculate option price
import numpy as np
from scipy.stats import norm
from numba import jit
from numba import int32, float32    # import the types
from numba.experimental import jitclass
from utils import characteristic_function_BS, f_tilde, characteristic_function_Heston, solve_system
from simulations import sim_gbm_paths, milstein_method_heston
import scipy.integrate as integrate


class BlackScholesMarket:
    def __init__(self, t, S0, S_t, r, sigma, T, K, M):
        self.t = t
        self.S0 = S0
        self.S_t = S_t
        self.r = r
        self.sigma = sigma
        self.T = T
        self.K = K
        self.M = M
        self.H = None
        self.d_1 = None
        self.d_2 = None
        self.d_calculator()

    # Black Scholes eu call formula
    def d_calculator(self):
        """Calculates d_1 and d_2 which is required for the Black-Scholes formula."""
        self.d_1 = (np.log(self.S_t / self.K) + (self.r + np.square(self.sigma) / 2) * (self.T - self.t)) / (self.sigma * np.sqrt(self.T - self.t))
        self.d_2 = self.d_1 - self.sigma * np.sqrt(self.T - self.t)

    def BlackScholes_EuCall(self):
        return self.S_t * norm.cdf(self.d_1) - self.K * np.exp(-self.r * (self.T - self.t)) * norm.cdf(self.d_2)
    
    # Black Scholes eu put formula
    def BlackScholes_EuPut(self):
        return self.K * np.exp(-self.r * (self.T - self.t)) * norm.cdf(-self.d_2) -  self.S_t * norm.cdf(-self.d_1)
    
    def BlackScholes_Eu_Barrier_Put(self):
        """Calculates the fair price for down and out put option in the Black-Scholes model."""
        # v1 = norm.cdf((np.log(self.H**2 / (self.K * self.S_t)) + (self.r+self.sigma**2 / 2) * (self.T - self.t)) / (self.sigma * np.sqrt(self.T - self.t)))
        # v2 = (self.H / self.S_t)**(1+2*self.r/self.sigma**2) * v1
        # v3 = np.exp(-self.r * (self.T - self.t)) * self.K * self.d_1
        # v4 = norm.cdf((np.log(self.H**2 / (self.K * self.S_t)) + (self.r-self.sigma**2 / 2) * (self.T - self.t)) / (self.sigma * np.sqrt(self.T - self.t)))
        # v5 = (self.H / self.S_t)**(2*self.r/self.sigma**2 - 1) * v4
        lambda_ = (self.r + self.sigma**2 / 2) * self.sigma**2
        y = np.log(self.H**2 / (self.S_t * self.K)) / (self.sigma * np.sqrt(self.T)) + lambda_ * self.sigma  * np.sqrt(self.T)
        x_1 = np.log(self.S_t / self.H) / (self.sigma * np.sqrt(self.T)) + lambda_ * self.sigma  * np.sqrt(self.T)
        y_1 = np.log(self.H / self.S_t) / (self.sigma * np.sqrt(self.T)) + lambda_ * self.sigma  * np.sqrt(self.T)

        v1 = self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d_2) - self.S0 * norm.cdf(-self.d_1) + self.S0 * norm.cdf(-x_1)
        v2 = self.K * np.exp(-self.r * self.T) * norm.cdf(-x_1 + self.sigma * np.sqrt(self.T)) - self.S0 * (self.H / self.S0)**(2 * lambda_) * (norm.cdf(y) - norm.cdf(y_1))
        v3 = self.K * np.exp(-self.r * self.T) * (self.H / self.S0)**(2 * lambda_ - 2) * (norm.cdf(y - self.sigma * np.sqrt(self.T)) - norm.cdf(y_1 - self.sigma * np.sqrt(self.T)))

        return v1 - v2 + v3
    
    def BlackScholes_Eu_Barrier_Call(self):
        """Calculates the fair price for down and out call option in the Black-Scholes model."""""
        v1 = norm.cdf((np.log(self.H**2 / (self.K * self.S_t)) + (self.r+self.sigma**2 / 2) * (self.T - self.t)) / (self.sigma * np.sqrt(self.T - self.t)))
        v2 = (self.H / self.S_t)**(1+2*self.r/self.sigma**2) * v1
        v3 = np.exp(-self.r * (self.T - self.t)) * self.K * norm.cdf((np.log(self.S_t / self.K) + (self.r - np.square(self.sigma) / 2) * (self.T - self.t)) / (self.sigma * np.sqrt(self.T - self.t)))
        v4 = norm.cdf((np.log(self.H**2 / (self.K * self.S_t)) + (self.r-self.sigma**2 / 2) * (self.T - self.t)) / (self.sigma * np.sqrt(self.T - self.t)))
        v5 = (self.H / self.S_t)**(2*self.r/self.sigma**2 - 1) * v4
        return self.S_t * (norm.cdf(self.d_1) - v2 - v3 - v5) 
        
    # Monte Carlo
    # def f(self, simulated_price, K):
        # return np.maximum(simulated_price - K, 0)

    # @jit(nopython=True, parallel=True)    
    def Eu_Option_BS_MC (self, excercise):
        simulation_array = np.zeros(shape=(self.M, ))
        for i in range(self.M):
            standard_normal_rv = np.random.randn()
            simulated_price = self.S0 * np.exp((self.r - np.square(self.sigma) / 2) * self.T + self.sigma * np.sqrt(self.T) * standard_normal_rv)
            simulated_option_price = np.maximum(simulated_price - self.K, 0) if excercise == "Call" else np.maximum(self.K - simulated_price, 0)
            simulation_array[i] = simulated_option_price
        simulation_array = simulation_array * np.exp(- self.r * self.T)
        simulated_mean_price = np.mean(simulation_array)
        # simulated_std = np.std(simulation_array, ddof=1)
        
        # Calculate 95% confidence interval
        # confidence_interval = [simulated_mean_price - 1.96 * simulated_std / np.sqrt(M), simulated_mean_price + 1.96 * simulated_std / np.sqrt(M)]
        return simulated_mean_price #, confidence_interval
    
    # @jit(nopython=True, parallel=True)
    def Eu_Option_BS_MC_Barrier(self, exercise_type, N=5_000):
        simulation_array = np.zeros(shape=(self.M, ))
        delta_t = self.T / N
        for i in range(self.M):
            simulated_path = np.zeros(shape=(N, ))
            simulated_path[0] = self.S0
            for j in range(1, N, 1):
                standard_normal_rv = np.random.randn()
                simulated_gbm = simulated_path[j - 1] * np.exp((self.r - np.square(self.sigma) / 2) * delta_t + self.sigma * standard_normal_rv * np.sqrt(delta_t))
                simulated_path[j] = simulated_gbm
                # check barrier:
                if simulated_gbm <= self.H:
                    simulation_array[i] = 0
                    break
                else:
                    simulation_array[i] = simulated_gbm
            # calculate payout
            simulation_array[i] = np.maximum(simulation_array[i] - self.K, 0) if exercise_type == "Call" else np.maximum(self.K - simulation_array[i], 0)
            
        simulation_array = simulation_array * np.exp(- self.r * self.T)
        simulated_mean_price = np.mean(simulation_array)

        return simulated_mean_price
    
    # PDE Methods
    def Eu_Option_BS_PDE(self, boundary_conditon = True):
        """PDE-method, currently with the implicit scheme.
        Returns the stock price matrix and the options prices at t=0."""
        # define parameters
        a = -0.7
        b = 0.4
        m = 500
        v_max = 2000
        q = 2*self.r / np.square(self.sigma)
        delta_x = (b-a)/m
        delta_t = np.square(self.sigma) * self.T / (2 * v_max)
        lambda_ = delta_t / np.square(delta_x)

        price_matrix = np.zeros(shape=(m + 1,m + 1))
        final_stock_prices = np.zeros(shape=(m + 1))
        w = np.zeros(shape=(m + 1))
        # Add stock prices for the first row:
        for i in range(m + 1):
            # price_matrix[m - i - 1, 0] = K * np.exp(a + i * delta_x)
            final_stock_prices[i] =  b - i * delta_x
            
        # calculate payout for the first row:
        for i in range(m + 1):
            x_i = final_stock_prices[i]
            wi = np.maximum(np.exp(x_i*0.5*(q+1)) - np.exp(x_i * 0.5 * (q-1)), 0)
            price_matrix[i, 0] = wi
            w[i] = wi
            
        A_impl = np.zeros(shape=(m + 1,m + 1))
        A_impl[0, 0] = 1 + 2*lambda_
        A_impl[0, 1] = - lambda_
        A_impl[-1, -2] = - lambda_
        A_impl[-1, -1] = 1 + 2*lambda_

        for i in range(m-1):
            A_impl[i + 1, i] = - lambda_
            A_impl[i + 1, i + 1] = 1 + 2 * lambda_
            A_impl[i + 1, i + 2] = - lambda_
        
        price_matrix[:, 0] = w
        A_impl_inv = np.linalg.inv(A_impl)
        # get the option prices
        for i in range(1, m + 1):
            t = i * delta_t
            x = final_stock_prices[0]
            price_matrix[:, i] = np.linalg.matrix_power(A_impl_inv, i) @ w
            # with boundary conditions
            if boundary_conditon == True: 
                price_matrix[0, i] = np.exp(0.5 * (q+1)*x + 0.25 * (q+1)**2 * t) - np.exp(0.5*(q-1)*x + 0.25*(q-1)**2 * t)
                price_matrix[-1, i] = 0
        
        # transform back
        S_matrix = np.zeros(shape=(m + 1))
        v_0 = np.zeros(shape=(m + 1))

        for i in range(m + 1):
            x_i = final_stock_prices[i]
            S_matrix[i] = self.K * np.exp(x_i)
            v_0[i] = self.K * price_matrix[i, -1] * np.exp(-x_i/2*(q-1)-0.5*self.sigma**2*self.T*(q+0.25*(q-1)**2))
        return S_matrix, v_0
    
    def Am_Option_BS_PDE(self, exercise_type, theta=0.5):
        # Compute delta_x, delta_t, x, lambda and set starting w
        a = -0.7
        b = 0.4
        m = 500
        nu_max = 2000

        q = (2 * self.r) / (self.sigma * self.sigma)
        delta_x = (b - a) / m
        delta_t = (self.sigma * self.sigma * self.T) / (2 * nu_max)
        lmbda = delta_t / (delta_x * delta_x)
        x = np.ones(m + 1) * a + np.arange(0, m + 1) * delta_x
        t = delta_t * np.arange(0, nu_max + 1)
        if exercise_type == "Call":
            g_nu = np.maximum(np.exp(x * 0.5 * (q + 1)) - np.exp(x * 0.5 * (q - 1)), np.zeros(m + 1))
        else:
            g_nu = np.maximum(np.exp(x * 0.5 * (q - 1)) - np.exp(x * 0.5 * (q + 1)), np.zeros(m + 1))
        w = g_nu[1:m]

        # Building matrix for t-loop
        lambda_theta = lmbda * theta
        diagonal = np.ones(m - 1) * (1 + 2 * lambda_theta)
        secondary_diagonal = np.ones(m - 2) * (- lambda_theta)
        b = np.zeros(m - 1)

        # t-loop as on p.77.
        for nu in range(0, nu_max):
            if exercise_type == "Call":
                g_nuPlusOne = np.exp((q + 1) * (q + 1) * t[nu + 1] / 4.0) * np.maximum(np.exp(x * 0.5 * (q + 1))
                                                                                    - np.exp(x * 0.5 * (q - 1)),
                                                                                    np.zeros(m + 1))
            else:
                g_nuPlusOne = np.exp((q + 1) * (q + 1) * t[nu + 1] / 4.0) * np.maximum(np.exp(x * 0.5 * (q - 1))
                                                                                    - np.exp(x * 0.5 * (q + 1)),
                                                                                    np.zeros(m + 1))
            b[0] = w[0] + lmbda * (1 - theta) * (w[1] - 2 * w[0] + g_nu[0]) + lambda_theta * g_nuPlusOne[0]
            b[1:m - 2] = w[1:m - 2] + lmbda * (1 - theta) * (w[2:m - 1] - 2 * w[1:m - 2] + w[0:m - 3])
            b[m - 2] = w[m - 2] + lmbda * (1 - theta) * (g_nu[m] - 2 * w[m - 2] + w[m - 3]) + lambda_theta * g_nuPlusOne[m]

            # Use Brennan-Schwartz algorithm to solve the linear equation system
            w = solve_system(diagonal, secondary_diagonal, secondary_diagonal, b, g_nuPlusOne[1:m])

            g_nu = g_nuPlusOne

        S = self.K * np.exp(x[1:m])
        v = self.K * w * np.exp(- 0.5 * x[1:m] * (q - 1) - 0.5 * self.sigma * self.sigma * self.T * ((q - 1) * (q - 1) / 4 + q))

        return S, v

    def integrand_Call(self, u):
        R = 1.2
        return np.exp(-self.r * self.T) / np.pi * np.real(f_tilde(self.K, R + 1j * u) * characteristic_function_BS(u - 1j * R, self.S0, self.r, self.sigma, self.T))
    
    def integrand_Put(self, u):
        R = -1.1
        return np.exp(-self.r * self.T) / np.pi * np.real(f_tilde(self.K, R + 1j * u) * characteristic_function_BS(u - 1j * R, self.S0, self.r, self.sigma, self.T))
    
    def Eu_Option_BS_LaPlace(self, exercise_type):
        if exercise_type == "Call":
            return integrate.quad(self.integrand_Call, 0, 50)[0] # returns V0
        else:
            return integrate.quad(self.integrand_Put, 0, 50)[0] # returns V0
    
    def Am_Option_BS_LS(self, exercise_type, N=20000):
        """Calculates the fair value of an american option using the Longstaff-Schwartz Algorithm."""
        # V0 = 0
        sim_pathes = sim_gbm_paths(self.S0, self.r, self.sigma, self.T, N, self.M)

        tau_n = np.ones(shape=N) * self.T
        optimal_stopping_time_idx = np.ones(shape=N, dtype=np.int32) * (self.M-1)

        # inner value
        if exercise_type == "Call":
            h = np.maximum(sim_pathes - self.K, 0)
        else:
            h = np.maximum(self.K - sim_pathes, 0)

        #terminal value
        V = h[:, -1]
        for i in range(self.M-1, 0, -1):
            t_i = i * self.T / self.M
            itm_pathes = np.where(h[:, i] > 0) # check if path is in the money

            if itm_pathes[0].size != 0: # check whether there are in the money pathes
                discounting_factor = np.exp(-self.r * (tau_n[itm_pathes] - t_i))
                
                # rg = np.polyfit(sim_pathes[itm_pathes, i].reshape(-1), V[itm_pathes] * discounting_factor, 5) #polynom fit
                rg_laguerre = np.polynomial.laguerre.lagfit(sim_pathes[itm_pathes, i].reshape(-1), V[itm_pathes] * discounting_factor, 4) #laguerre fit
                # linear regression
                # A = np.vstack([sim_pathes[itm_pathes, i].reshape(-1), np.ones(len(sim_pathes[itm_pathes, i].reshape(-1)))]).T
                # m, c = np.linalg.lstsq(A, V[itm_pathes] * discounting_factor, rcond=None)[0]
                
                # poly_values = np.polyval(rg, sim_pathes[itm_pathes, i].reshape(-1)) # predict payout next period
                laguerre_values = np.polynomial.laguerre.lagval(sim_pathes[itm_pathes, i].reshape(-1), rg_laguerre)
                # linear_regression_values = m * sim_pathes[itm_pathes, i].reshape(-1) + c
                
                # idx = np.where(h[itm_pathes, i] >= poly_values)[1]
                idx = np.where(h[itm_pathes, i] >= laguerre_values)[1]
                # idx = np.where(h[itm_pathes, i] >= linear_regression_values)[1]
                
                tau_n[idx] = t_i
                optimal_stopping_time_idx[idx] = i - 1
                V[idx] = h[idx, i]
    

        x_coords = np.arange(N)
        discounting = np.exp(-self.r * tau_n)
        V0 = np.maximum(h[0, 0], np.mean(discounting * h[x_coords, optimal_stopping_time_idx]))
        return V0


class HestonModel():
    def __init__(self, t, S0, S_t, r, sigma_tilde, T, K, M, lambda_parameter, m, kappa, gamma0):
        self.t = t
        self.S0 = S0
        self.S_t = S_t
        self.r = r
        self.sigma_tilde = sigma_tilde
        self.T = T
        self.K = K
        self.M = M
        self.H = None
        self.lambda_parameter = lambda_parameter
        self.m = m
        self.kappa = kappa
        self.gamma0 = gamma0

    #jit(nopython=True) 
    def Heston_EuCall_MC_Euler (self, exercise_type):
        """Heston model calculation vanilla options with Monte-Carlo method and Euler method for the SDE."""
        call_prices = np.zeros(shape=(self.M))
        
        for z in range(self.M):
            # Initialise
            S = np.zeros(shape=(self.m, ))
            Gamma = np.zeros(shape=(self.m, ))
            Gamma[0] = self.gamma0
            delta_t = self.T /self.m
            S[0] = self.S0
            for i in range(1, self.m , 1):
                # volatility modeling
                delta_W_tilde = np.random.standard_normal() * np.sqrt(delta_t)
                # avoid negative values for Gamma: take maximum of Gamma[i - 1] and 0
                Gamma[i] = np.maximum(Gamma[i - 1], 0) + (self.kappa - self.lambda_parameter * np.maximum(Gamma[i - 1], 0) ) * delta_t + np.sqrt(np.maximum(Gamma[i - 1], 0)) * self.sigma_tilde * delta_W_tilde
                
                # stock price modeling
                delta_W_Q = np.random.standard_normal() * np.sqrt(delta_t)
                # avoid negative values for Gamma: take maximum of Gamma[i - 1] and 0
                S[i] = S[i - 1]  + S[i - 1] * self.r * delta_t + S[i - 1] * np.sqrt(np.maximum(Gamma[i], 0)) * delta_W_Q
            
            # calculate Payout at t = T
            payout = np.maximum(S[-1] - self.K, 0) if exercise_type == "Call" else np.maximum(self.K - S[-1], 0)
            call_prices[z] = payout
            
        # monte carlo estimate + discounting
        V0 = 1 / self.M * np.sum(call_prices) * np.exp(-self.r * self.T)
        
        # # calculate confidence interval
        # simulated_mean_price = np.mean(call_prices)
        # simulated_std = np.std(call_prices)
        # confidence_interval = [simulated_mean_price - 1.96 * simulated_std / np.sqrt(M), simulated_mean_price + 1.96 * simulated_std / np.sqrt(M)] 
        return V0
    
    def integrand_Call(self, u):
        R = 1.2
        return np.exp(-self.r * self.T) / np.pi * np.real(f_tilde(self.K, R + 1j * u) * characteristic_function_Heston(u - 1j * R, self.S0, self.r, self.sigma_tilde, self.lambda_parameter, self.kappa, self.gamma0, self.T))
    
    def integrand_Put(self, u):
        R = -1.1
        return np.exp(-self.r * self.T) / np.pi * np.real(f_tilde(self.K, R + 1j * u) * characteristic_function_Heston(u - 1j * R, self.S0, self.r, self.sigma_tilde, self.lambda_parameter, self.kappa, self.gamma0, self.T))
    
    def Heston_EuCall_MC_LaPlace(self, exercise_type):
        if exercise_type == "Call":
            return integrate.quad(self.integrand_Call, 0, 50)[0] # returns V0
        else:
            return integrate.quad(self.integrand_Put, 0, 50)[0] # returns V0
    
    def Am_Option_Heston_LS(self, exercise_type, N=1000):
        """Calculates the fair value of an american option using the Longstaff-Schwartz Algorithm in the heston model."""
        sim_pathes = milstein_method_heston(self.S0, self.gamma0, self.r, self.sigma_tilde, self.kappa, self.lambda_parameter, self.T, self.m, N)

        tau_n = np.ones(shape=N) * self.T
        optimal_stopping_time_idx = np.ones(shape=N, dtype=np.int32) * (self.m-1)

        # inner value
        if exercise_type == "Call":
            h = np.maximum(sim_pathes - self.K, 0)
        else:
            h = np.maximum(self.K - sim_pathes, 0)

        #terminal value
        V = h[:, -1]
        for i in range(self.m-1, 0, -1):
            t_i = i * self.T / self.m
            itm_pathes = np.where(h[:, i] > 0) # check if path is in the money

            if itm_pathes[0].size != 0: # check whether there are in the money pathes
                discounting_factor = np.exp(-self.r * (tau_n[itm_pathes] - t_i))
                
                # rg = np.polyfit(sim_pathes[itm_pathes, i].reshape(-1), V[itm_pathes] * discounting_factor, 5) #polynom fit
                rg_laguerre = np.polynomial.laguerre.lagfit(sim_pathes[itm_pathes, i].reshape(-1), V[itm_pathes] * discounting_factor, 4) #laguerre fit
                # linear regression
                # A = np.vstack([sim_pathes[itm_pathes, i].reshape(-1), np.ones(len(sim_pathes[itm_pathes, i].reshape(-1)))]).T
                # m, c = np.linalg.lstsq(A, V[itm_pathes] * discounting_factor, rcond=None)[0]
                
                # poly_values = np.polyval(rg, sim_pathes[itm_pathes, i].reshape(-1)) # predict payout next period
                laguerre_values = np.polynomial.laguerre.lagval(sim_pathes[itm_pathes, i].reshape(-1), rg_laguerre)
                # linear_regression_values = m * sim_pathes[itm_pathes, i].reshape(-1) + c
                
                # idx = np.where(h[itm_pathes, i] >= poly_values)[1]
                idx = np.where(h[itm_pathes, i] >= laguerre_values)[1]
                # idx = np.where(h[itm_pathes, i] >= linear_regression_values)[1]
                
                tau_n[idx] = t_i
                optimal_stopping_time_idx[idx] = i - 1
                V[idx] = h[idx, i]
    

        x_coords = np.arange(N)
        discounting = np.exp(-self.r * tau_n)
        V0 = np.maximum(h[0, 0], np.mean(discounting * h[x_coords, optimal_stopping_time_idx]))
        return V0
