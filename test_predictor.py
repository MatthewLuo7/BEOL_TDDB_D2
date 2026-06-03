import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

class WaferReliabilityPredictor:
    def __init__(self, weibull_model, M_structures: float = 1e6, F_target: float = 1e-4):
        """
        :param weibull_model: 已经加载并训练好的 WeibullGPR 模型实例
        :param M_structures: 每个 Die 包含的该类结构总数 (M)
        :param F_target: 目标相对可靠性失效阈值 (如 0.01% 即 1e-4)
        """
        self.weibull_model = weibull_model
        self.M = M_structures
        self.F_target = F_target
        
    def _train_wafer_gpr(self, x, y, values):
        """训练单特征的空间插值 GPR 模型"""
        X_train = np.column_stack([x, y])
        kernel = ConstantKernel(1.0) * RBF(length_scale=3.0) + WhiteKernel(noise_level=0.05)
        gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        gpr.fit(X_train, values)
        return gpr

    def _generate_die_samples(self, valid_dies_x, valid_dies_y, N_samples_per_dim: int):
        """
        在每个有效的 Die 内部均匀抽样 N_samples_per_dim * N_samples_per_dim 个点
        返回形状为 (D * N, 2) 的坐标矩阵，其中 D 为 Die 数量，N 为每个 Die 的抽样数
        """
        D = len(valid_dies_x)
        N = N_samples_per_dim ** 2
        
        # 在 [-0.5, 0.5] 的 Die 范围内生成网格点
        offset_lin = np.linspace(-0.45, 0.45, N_samples_per_dim)
        ox, oy = np.meshgrid(offset_lin, offset_lin)
        ox, oy = ox.ravel(), oy.ravel() # shape: (N,)
        
        # 广播相加：将局部偏移叠加到中心坐标上
        # X_coords shape: (D, N)
        X_coords = valid_dies_x[:, np.newaxis] + ox[np.newaxis, :]
        Y_coords = valid_dies_y[:, np.newaxis] + oy[np.newaxis, :]
        
        return X_coords.flatten(), Y_coords.flatten(), D, N

    def _solve_lifetime_vectorized(self, beta_matrix, eta_matrix, max_iter=50, tol=1e-6):
        """
        使用完全向量化的 Newton-Raphson 法，同时解出所有 Die 的寿命 t。
        beta_matrix, eta_matrix: shape (D, N)
        """
        D, N = beta_matrix.shape
        C = -np.log(1.0 - self.F_target) # 常数项
        
        # 初始猜测 t_0: 使用均值近似解
        mean_beta = np.mean(beta_matrix, axis=1, keepdims=True)
        mean_eta = np.mean(eta_matrix, axis=1, keepdims=True)
        t = mean_eta * (C / self.M) ** (1.0 / mean_beta) # shape: (D, 1)
        
        # 牛顿法迭代
        for _ in range(max_iter):
            # term 形状为 (D, N)，利用广播机制
            term = (t / eta_matrix) ** beta_matrix 
            
            # 评估函数 f(t) = 0
            f_t = (self.M / N) * np.sum(term, axis=1, keepdims=True) - C
            
            # 如果最大的残差小于容差，则停止迭代
            if np.max(np.abs(f_t)) < tol:
                break
                
            # 评估导数 f'(t)
            f_prime_t = (self.M / N) * np.sum((beta_matrix / t) * term, axis=1, keepdims=True)
            
            # 更新 t
            t = t - f_t / f_prime_t
            
            # 防御性编程：防止出现非法的负数时间
            t = np.maximum(t, 1e-12) 
            
        return t.flatten() # shape: (D,)

    def predict_wafer_reliability(self, 
                                  x_die, y_die, 
                                  vl_measured, ll_measured, 
                                  N_samples_per_dim=10):
        """
        主执行管线
        :param x_die, y_die: 测量的有效 Die 坐标 (一维数组)
        :param vl_measured, ll_measured: 对应 Die 上测量的 vl 和 ll 值
        :param N_samples_per_dim: 每个 Die 内部在 x 和 y 方向的抽样点数 (默认 10*10=100点)
        """
        print("1. Training spatial GPR for vl and ll...")
        gpr_vl = self._train_wafer_gpr(x_die, y_die, vl_measured)
        gpr_ll = self._train_wafer_gpr(x_die, y_die, ll_measured)
        
        print(f"2. Generating {N_samples_per_dim**2} sample points per die...")
        xx_samples, yy_samples, D, N = self._generate_die_samples(x_die, y_die, N_samples_per_dim)
        sample_coords = np.column_stack([xx_samples, yy_samples])
        
        print("3. Interpolating vl and ll at sub-die level...")
        vl_pred = gpr_vl.predict(sample_coords)
        ll_pred = gpr_ll.predict(sample_coords)
        
        print("4. Predicting Weibull parameters (beta, eta)...")
        # 组合输入并通过你的 WeibullGPR 模型
        X_weibull = np.column_stack([vl_pred, ll_pred])
        
        # 这里的 predict 已经内嵌了你之前配置好的 Scale / Clip 层
        beta_pred, eta_pred = self.weibull_model.predict(X_weibull, verbose=False) 
        
        # 重整为 (D, N) 形状，准备进入方程求解
        beta_matrix = beta_pred.reshape((D, N))
        eta_matrix = eta_pred.reshape((D, N))
        
        print("5. Solving weakest-link lifetime equation using vectorized Newton-Raphson...")
        t_lifetimes = self._solve_lifetime_vectorized(beta_matrix, eta_matrix)
        
        print("Done!")
        # 返回 Die 坐标及对应的可靠性分数 (寿命 t)
        return x_die, y_die, t_lifetimes

if __name__ == "__main__":
    from Code.local_percolation_gpr.weibull_gpr import load_model
    from wafer_plot import load_matrix_csv, matrix_to_sparse_points

    weibull_model = load_model(actual_vl_max=25.0, actual_ll_max=12.5)

    predictor = WaferReliabilityPredictor(
        weibull_model=weibull_model, 
        M_structures=1_000_000, 
        F_target=1e-4
    )

    vl_matrix = load_matrix_csv('data/lot_009/csv/wafer_14/Space.csv')
    ll_matrix = load_matrix_csv('data/lot_009/csv/wafer_14/MS.csv')
    x, y, vl_val = matrix_to_sparse_points(vl_matrix)
    _, _, ll_val = matrix_to_sparse_points(ll_matrix)

    x_die, y_die, t_scores = predictor.predict_wafer_reliability(
        x_die=x, y_die=y, 
        vl_measured=vl_val, ll_measured=ll_val, 
        N_samples_per_dim=32  # 意味着每个 Die 内部产生 10x10=100 个抽样点
    )