import pathlib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

default_path = pathlib.Path('./Code/local_percolation_gpr/simulated_data/')
via_dim_y = 10.5
via_dim_z = 21.0
line_dim_x = 10.5
line_dim_y = 21.0
line_dim_z = 21.0
radius_N = 2.0

def load_model(
        m_param:float=5.0,
        radius:float=0.45,
        train_data_path=None,
        actual_vl_max:float=25.0,  # The actual maximum value of vl in the input data
        actual_ll_max:float=12.5   # The actual maximum value of ll in the input data
    ):
    # Only these parameters are simulated
    assert m_param in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert radius in [0.45, 0.65]

    if train_data_path is not None: 
        data = np.load(train_data_path) 
    else:
        dir_name = f'vy{via_dim_y:.2f}_vz{via_dim_z:.2f}_lx{line_dim_x:.2f}_ly{line_dim_y:.2f}_lz{line_dim_z:.2f}_r{radius:.2f}'
        data = np.load(default_path / dir_name / f"weibull_gpr_m{m_param:.2f}_rN{radius_N:.2f}.npz")

    X = data["X"]
    beta = data["beta"]
    eta = data["eta"]

    # Instantiate the model and pass the actual maximum values for scaling
    model = WeibullGPR(actual_vl_max=actual_vl_max, actual_ll_max=actual_ll_max)
    model.fit(X=X, beta=beta, eta=eta)

    return model

class WeibullGPR:
    def __init__(self, 
                 actual_vl_max: float = 25.0, 
                 actual_ll_max: float = 12.5, 
                 vl_bounds: tuple = (0.0, 25.0), 
                 ll_bounds: tuple = (5.0, 12.5)):
        """
        Initialize the model and automatically compute parameters 
        for the preprocessing layer (Scale + Clip).
        """
        self.vl_bounds = vl_bounds
        self.ll_bounds = ll_bounds
        
        # Auto-calculate Scale: model space max bound / actual input max
        # A small epsilon (1e-9) is added to prevent division by zero
        self.vl_scale = self.vl_bounds[1] / max(actual_vl_max, 1e-9)
        self.ll_scale = self.ll_bounds[1] / max(actual_ll_max, 1e-9)

        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3)) *
            RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
        )

        self.gp_beta = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5
        )

        self.gp_eta = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5
        )

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        """
        Core scaling and clipping layer.
        """
        # Create a copy to prevent modifying the original array in place
        X_processed = np.copy(X)
        
        # 1. Scaling
        X_processed[:, 0] *= self.vl_scale
        X_processed[:, 1] *= self.ll_scale
        
        # 2. Clipping
        X_processed[:, 0] = np.clip(X_processed[:, 0], self.vl_bounds[0], self.vl_bounds[1])
        X_processed[:, 1] = np.clip(X_processed[:, 1], self.ll_bounds[0], self.ll_bounds[1])
        
        return X_processed

    def fit(self, X:np.ndarray,
            beta:np.ndarray,
            eta:np.ndarray,
            train_save_path:str=None):
        """
        X [:, 2]
        (vl_space, ll_space)
        Note: Assumes the training data passed to fit already aligns with 
        the 0-25 and 5-12.5 bounds, thus requiring no scaling.
        """
        # Log-domain to stabilize values and avoid negative values
        self.gp_beta.fit(X, np.log(beta))
        self.gp_eta.fit(X, np.log(eta))

        if train_save_path is not None:
            np.savez(
                train_save_path,
                X=X,
                beta=beta,
                eta=eta
            )

    def predict(self, X:np.ndarray, verbose:bool=False, conf_factor:float=2.0):
        # Apply Scale and Clip processing before inference
        X_processed = self._preprocess(X)

        if verbose:
            log_beta, std_beta = self.gp_beta.predict(X_processed, return_std=True)
            log_eta,  std_eta  = self.gp_eta.predict(X_processed, return_std=True)

            log_beta = np.expand_dims(log_beta, axis=-1)
            std_beta = np.expand_dims(std_beta, axis=-1)
            log_eta = np.expand_dims(log_eta, axis=-1)
            std_eta = np.expand_dims(std_eta, axis=-1)

            log_beta_lower = log_beta - std_beta * conf_factor
            log_beta_upper = log_beta + std_beta * conf_factor
            log_eta_lower = log_eta - std_eta * conf_factor
            log_eta_upper = log_eta + std_eta * conf_factor

            return (
                np.exp(np.stack([log_beta, log_beta_lower, log_beta_upper], axis=-1)),
                np.exp(np.stack([log_eta, log_eta_lower, log_eta_upper], axis=-1)),
            )
        else:
            # Replaced the undefined Xq with the preprocessed X_processed
            return (
                np.exp(self.gp_beta.predict(X_processed)),
                np.exp(self.gp_eta.predict(X_processed))
            )



# import pathlib
# import numpy as np
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

# default_path = pathlib.Path('./Code/local_percolation_gpr/simulated_data/')
# via_dim_y = 10.5
# via_dim_z = 21.0
# line_dim_x = 10.5
# line_dim_y = 21.0
# line_dim_z = 21.0
# radius_N = 2.0

# def load_model(
# 		m_param:float=5.0,
# 		radius:float=0.45,
# 		train_data_path=None):
# 	# only these parameters are simulated
# 	assert m_param in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,, 7.0, 8.0, 9.0, 10.0]
# 	assert radius in [0.45, 0.65]

# 	if train_data_path is not None:	
# 		data = np.load(train_save_path)
# 	else:
# 		dir_name = f'vy{via_dim_y:.2f}_vz{via_dim_z:.2f}_lx{line_dim_x:.2f}_ly{line_dim_y:.2f}_lz{line_dim_z:.2f}_r{radius:.2f}'
# 		data = np.load(default_path / dir_name / f"weibull_gpr_m{m_param:.2f}_rN{radius_N:.2f}.npz")

# 	X = data["X"]
# 	beta = data["beta"]
# 	eta = data["eta"]

# 	model = WeibullGPR()
# 	model.fit(X=X, beta=beta, eta=eta)

# 	return model

# class WeibullGPR:
# 	def __init__(self):
# 		kernel = (
# 			ConstantKernel(1.0, (1e-3, 1e3)) *
# 			RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
# 			+ WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
# 		)

# 		self.gp_beta = GaussianProcessRegressor(
# 			kernel=kernel,
# 			n_restarts_optimizer=5
# 		)

# 		self.gp_eta = GaussianProcessRegressor(
# 			kernel=kernel,
# 			n_restarts_optimizer=5
# 		)

# 	def fit(self, X:np.ndarray,
# 			beta:np.ndarray,
# 			eta:np.ndarray,
# 			train_save_path:str=None):
# 		"""
# 		X [:, 2]
# 		(vl_space, ll_space)
# 		"""
# 		# log-domain to stablize values and avoid negative values
# 		self.gp_beta.fit(X, np.log(beta))
# 		self.gp_eta.fit(X, np.log(eta))

# 		if train_save_path is not None:
# 			np.savez(
# 				train_save_path,
# 				X=X,
# 				beta=beta,
# 				eta=eta
# 			)

# 	def predict(self, X:np.ndarray, verbose:bool=False, conf_factor:float=2.):
# 		if verbose:
# 			log_beta, std_beta = self.gp_beta.predict(X, return_std=True)
# 			log_eta,  std_eta  = self.gp_eta.predict(X, return_std=True)

# 			log_beta = np.expand_dims(log_beta, axis=-1)
# 			std_beta = np.expand_dims(std_beta, axis=-1)
# 			log_eta = np.expand_dims(log_eta, axis=-1)
# 			std_eta = np.expand_dims(std_eta, axis=-1)

# 			log_beta_lower = log_beta - std_beta * conf_factor
# 			log_beta_upper = log_beta + std_beta * conf_factor
# 			log_eta_lower = log_eta - std_eta * conf_factor
# 			log_eta_upper = log_eta + std_eta * conf_factor

# 			return (
# 				np.exp(np.stack([log_beta, log_beta_lower, log_beta_upper], axis=-1)),
# 				np.exp(np.stack([log_eta, log_eta_lower, log_eta_upper], axis=-1)),
# 			)
# 		else:
# 			return (
# 				np.exp(self.gp_beta.predict(Xq)),
# 				np.exp(self.gp_eta.predict(Xq))
# 			)