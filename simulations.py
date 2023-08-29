import numpy as np

def sim_gbm_paths(S0, r, sigma, T, N, m):
    """Simulates N different geometric brownian motion pathes with m equidistant pathes."""
    sim_pathes = np.zeros(shape=(N, m))
    sim_pathes[:, 0] = S0
    dt = T / m
    for i in range(N):
        brownian_motion = np.random.normal(loc=0.0, scale=1.0, size=m-1) 
        sim_pathes[i, 1:] = S0 * np.cumproduct(np.exp((r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * brownian_motion))
    
    return sim_pathes

def euler_method_heston(X0, gamma0, r, sigma, kappa, theta, T, m, N):
    """Simulates N different pathes  in the heston model with m equidistant pathes using the euler method."""
    delta_t = T/m
    Y_array = np.zeros(shape=(N, m))
    Gamma = np.zeros(shape=(N, m))
    
    Y_array[:, 0] = X0
    Gamma[: , 0] = gamma0
    for j in range(N):
        W_1 = np.random.normal(loc=0.0, scale=1.0, size=m)
        W_2 = np.random.normal(loc=0.0, scale=1.0, size=m)
        
        for i in range(1, m):
            # avoid negative values for Gamma: take maximum of Gamma[i - 1] and 0
            Gamma[j, i] = np.maximum(Gamma[j, i - 1], 0) + (kappa - theta * np.maximum(Gamma[j, i - 1], 0) ) * delta_t + np.sqrt(np.maximum(Gamma[j, i - 1], 0)) * sigma * np.sqrt(delta_t) * W_2[i]

            # avoid negative values for Gamma: take maximum of Gamma[i - 1] and 0
            Y_array[j, i] = Y_array[j, i - 1]  + Y_array[j, i - 1] * r * delta_t + Y_array[j, i - 1] * np.sqrt(np.maximum(Gamma[j, i], 0)) * np.sqrt(delta_t) * W_1[i]
            
    return Y_array         
            
def milstein_method_heston(X0, gamma0, mu, sigma, kappa, theta, T, m, N):
    """Simulates N different pathes  in the heston model with m equidistant pathes using the milstein method."""
    delta_t = T/m
    Y_array = np.zeros(shape=(N, m))
    gamma_array = np.zeros(shape=(N, m))
    
    Y_array[:, 0] = X0
    gamma_array[:, 0] = gamma0
    for j in range(N):
        W_1 = np.random.normal(loc=0.0, scale=1.0, size=m) #  brownian motion for stock for stock process
        W_2 = np.random.normal(loc=0.0, scale=1.0, size=m) # brownian motion for volatility process
        
        for i in range(1, m):
            gamma_array[j, i] = np.maximum(gamma_array[j, i - 1], 0) + kappa *(theta - np.maximum(gamma_array[j, i - 1], 0))*delta_t + sigma * np.sqrt(delta_t * np.maximum(gamma_array[j, i - 1], 0)) * W_2[i] + 0.25 * sigma**2 * (W_2[i]**2 - 1)
            Y_array[j, i] = Y_array[j, i-1] + Y_array[j, i-1] * (mu * delta_t + np.sqrt(np.maximum(gamma_array[j, i], 0)) * W_1[i] * np.sqrt(delta_t))

    return Y_array