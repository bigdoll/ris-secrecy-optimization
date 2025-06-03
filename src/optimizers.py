import numpy as np
import cvxpy as cp
import time
import logging
from utils import Utils
from gamma_utils import GammaUtils
from power_utils import PowerUtils
from typing import Callable, Tuple, Optional, Dict
from scipy.optimize import minimize_scalar
from collections import defaultdict
        
logger = logging.getLogger(__name__)

# def _line_search(func: Callable[[float], float],
#                  lo: float,
#                  hi: float,
#                  tol: float = 1e-3) -> float:
#     """
#     Wrapper to SciPy's bounded scalar minimization (Brent's method) for maximization.
#     Converts maximization to minimization of -func.
#     """
#     res = minimize_scalar(lambda x: -func(x), bounds=(lo, hi), method='bounded', options={'xatol': tol})
#     return res.x

def _line_search(func: Callable[[float], float],
                 lo: float,
                 hi: float,
                 tol: float = 1e-3) -> float:
    """
    Perform 1D maximization over [lo, hi] by minimizing -func using Brent's method.
    """
    res = minimize_scalar(lambda x: -func(x),
                          bounds=(lo, hi),
                          method='bounded',
                          options={'xatol': tol})
    return res.x

class Optimizer:
    def __init__(self, H, G_B, G_E, GE_error, sigma_sq, sigma_RIS_sq, sigma_e_sq, mu, Pc, PRmax, a, Ptmax, BW, scsi_bool=0, utils_cls=Utils, gamma_utils_cls=GammaUtils, power_utils_cls=PowerUtils):
        """
        Initialize the Optimizer class with given parameters.

        Parameters:
        - H: Channel matrix from UEs to RIS.
        - G_B: Channel matrix from RIS to Bob.
        - g_E: Channel matrix from RIS to Eve.
        - sigma_sq: Noise power variance at the receiver.
        - sigma_RIS_sq: Noise power variance at RIS.
        - sigma_g_sq: Noise variance for Eve's channel estimation error.
        - mu: Amplifier inefficiency factor.
        - Pc: Static power consumption.
        - a: Maximum reflection amplitude.
        - Pmax: Maximum power.
        - BW: Bandwidth.
        - scsi_bool: Boolean indicating whether to consider channel state information (CSI).
        - utils_cls: Utility class for common functions.
        - gamma_utils_cls: Utility class for gamma-related functions.
        - power_utils_cls: Utility class for power-related functions.
        """
        self.H = H
        self.G_B = G_B
        self.G_E = G_E if scsi_bool == 1 else  G_E + GE_error # G_E - GE_error if scsi_bool == 1 else G_E
        self.GE_error = GE_error
        self.sigma_sq = sigma_sq
        self.sigma_RIS_sq = sigma_RIS_sq
        self.sigma_e_sq = sigma_e_sq
        self.mu = mu
        self.Pc = Pc
        self.PRmax = PRmax
        self.a = a
        self.Ptmax = Ptmax
        self.BW = BW
        self.scsi_bool = scsi_bool
        self.utils_cls = utils_cls
        self.gamma_utils_cls = gamma_utils_cls(self.H, self.G_B, self.G_E, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.mu, self.Pc, self.scsi_bool) 
        self.power_utils_cls = power_utils_cls(self.H, self.G_B, self.G_E, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.mu, self.Pc, self.scsi_bool)
    
    def optimize_with_quantization(self, gamma_sol, p_sol, ris_state, a_min, a_max, bits_range):
        quantization_results = {
            'Gamma': {},
            'SR': {},
            'SSR': {},
            'GEE': {},
            'SEE': {}
        }
        GE_true = self.G_E + self.GE_error if self.scsi_bool == 1 else self.G_E # self.G_E + self.GE_error if self.scsi_bool == 1 else self.G_E 
        
        for bits in bits_range:
            bits_phase, bits_amplitude = bits
            print(f"Quantizing with bits_phase={bits_phase} and bits_amplitude={bits_amplitude}")
            gamma_sol_Q = self.utils_cls.project_to_quantized_levels(gamma_sol, a_min,  a_max, bits_phase, bits_amplitude)
            
            sr_sol_Q = self.utils_cls.SR_active_algo1(self.G_B, self.H, gamma_sol_Q, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.scsi_bool, True, "Bob");
            ssr_sol_Q = max(
                self.utils_cls.SR_active_algo1(self.G_B, self.H, gamma_sol_Q, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.scsi_bool, True, "Bob") -
                self.utils_cls.SR_active_algo1(GE_true, self.H, gamma_sol_Q, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.scsi_bool, True, "Eve"), 0
            )
            
            gee_sol_Q =  self.BW * self.utils_cls.GEE_active_algo1(self.G_B, self.H, gamma_sol_Q, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.scsi_bool, True, "Bob");
            see_sol_Q = self.BW * max(
                self.utils_cls.GEE_active_algo1(self.G_B, self.H, gamma_sol_Q, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.scsi_bool, True, "Bob") -
                self.utils_cls.GEE_active_algo1(GE_true, self.H, gamma_sol_Q, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.scsi_bool, True, "Eve"), 0
            )

            quantization_results['Gamma'][(bits_phase, bits_amplitude)] = gamma_sol_Q
            quantization_results['SR'][(bits_phase, bits_amplitude)] = sr_sol_Q
            quantization_results['SSR'][(bits_phase, bits_amplitude)] = ssr_sol_Q
            quantization_results['GEE'][(bits_phase, bits_amplitude)] = gee_sol_Q
            quantization_results['SEE'][(bits_phase, bits_amplitude)] = see_sol_Q
            
            # quantization_results[(bits_phase, bits_amplitude)] = {                'Gamma': gamma_sol_Q,
            #     'SSR': ssr_sol_Q,
            #     'SEE': see_sol_Q
            # }
        
        return quantization_results

    # def gamma_cvxopt_algo1_mod(self, gamma, p, opt_bool, rf_state, ris_state, cons_state):
    #     """
    #     Optimize gamma using the CVX optimization algorithm.

    #     Parameters:
    #     - gamma: Initial reflection coefficients.
    #     - p: Power allocation vector for UEs.
    #     - opt_bool: Boolean indicating whether to optimize energy efficiency.
    #     - ris_state: State of RIS ('active' or 'passive').
    #     - cons_state: Constraint state ('global' or 'local').

    #     Returns:
    #     - gamma_sol: Optimized reflection coefficients.
    #     - SSR_approx_sol: Secrecy sum rate approximation solution.
    #     - SEE_approx_sol: Secrecy energy efficiency approximation solution.
    #     - iter: Number of iterations.
    #     - tcpx_gamma: Time complexity for gamma optimization.
    #     """
    #     ris_bool = 1 if ris_state == 'active' else 0
    #     iter = 0
    #     gamma_sol = gamma.copy()
    #     gamma_0 = gamma.copy()
    #     N = gamma.shape[0]
    #     SSR_approx_prev = 0
    #     SEE_approx_prev = 0
    #     ln_prev = 0

    #     R = self.utils_cls.compute_R(self.H, p, self.sigma_RIS_sq)
    #     Rn = np.linalg.norm(R, 'fro')
    #     Rnorm = R / Rn
        
    #     if ris_state == 'active':
    #         PRmax = (self.a - 1)* np.real(np.trace(Rnorm)) if rf_state == 'RF-Gain' else self.PRmax
    #     else:
    #         PRmax = 0
    #     # PRmax = self.PRmax / Rn
        
    #     # if ris_state == 'active':
    #     #     PRmax = self.a * np.real(np.trace(Rnorm))
    #     # else:
    #     #     PRmax = 0

    #     SR_approx_nxt_Bob = self.gamma_utils_cls.SR_active_concave_gamma_Bob(gamma_sol, gamma_0, p, cvx_bool=1)
    #     SR_approx_nxt_Eve = self.gamma_utils_cls.SR_active_concave_gamma_Eve(gamma_sol, gamma_0, p, cvx_bool=1)
    #     SSR_approx_nxt = max(SR_approx_nxt_Bob.value + SR_approx_nxt_Eve.value, 0)
    #     Pc_eq = self.gamma_utils_cls.compute_Pc_eq_gamma(p)
    #     SEE_approx_nxt = SSR_approx_nxt / (ris_bool * np.real(np.trace(R @ (gamma_sol @ gamma_sol.conj().T))) + Pc_eq)

    #     ln_nxt = SSR_approx_nxt if opt_bool == 0 else SEE_approx_nxt
        
    #     # if opt_bool == 0: 
    #     #     ln_nxt = SSR_approx_nxt
    #     # else:
    #     #     ln_nxt = SEE_approx_nxt

    #     status = "unknown"
    #     tol = 1e-3
    #     flag = False

    #     if ln_nxt < tol:
    #         temp = ln_nxt
    #         ln_nxt = 10 * tol
    #         flag = True

    #     start_time = time.time()

    #     while ln_nxt - ln_prev > tol or (status not in ["optimal", "optimal_inaccurate"]):
    #         if flag:
    #             ln_nxt = temp
    #             flag = False

    #         iter += 1
    #         SSR_approx_prev = SSR_approx_nxt
    #         SEE_approx_prev = SEE_approx_nxt
    #         ln_prev = ln_nxt
    #         gamma_0 = gamma_sol.copy()

    #         gamma_opt = cp.Variable((N, 1), complex=True)
    #         objective = self.gamma_utils_cls.SR_active_concave_gamma_Bob(gamma_opt, gamma_0, p, cvx_bool=0) + \
    #                     self.gamma_utils_cls.SR_active_concave_gamma_Eve(gamma_opt, gamma_0, p, cvx_bool=0) - \
    #                     ris_bool * opt_bool * ln_prev * cp.real(cp.sum(cp.quad_form(gamma_opt, R)))

    #         if ris_state == 'active':
    #             constraints = [
    #                 cp.real(gamma_0.conj().T @ Rnorm @ gamma_0) + 2 * cp.real(gamma_0.conj().T @ Rnorm @ (gamma_opt - gamma_0)) >= cp.real(cp.trace(Rnorm)),
    #                 cp.inv_pos(PRmax) * cp.real(cp.quad_form(gamma_opt, R)) <= cp.inv_pos(PRmax) * PRmax + cp.inv_pos(PRmax) * cp.real(cp.trace(R)) if rf_state == "RF-Power" else cp.real(cp.quad_form(gamma_opt, Rnorm)) <= PRmax + cp.real(cp.trace(Rnorm))
    #             ]
    #         else:
    #             if cons_state == "global":
    #                 constraints = [cp.real(cp.quad_form(gamma_opt, Rnorm)) <= cp.real(cp.trace(Rnorm))]
    #             else:
    #                 constraints = [cp.square(cp.abs(gamma_opt)) <= 1] # self.a
            
    #         problem = cp.Problem(cp.Maximize(objective), constraints)
    #         # problem.solve(solver='MOSEK', warm_start=True)

    #         try:
    #             problem.solve(solver='MOSEK', warm_start=True)
    #         except cp.error.SolverError as e:
    #             print("MOSEK failed, trying another solver. Error:", e)
    #             problem.solve(solver='SCS', warm_start=True)

    #         gamma_sol = gamma_opt.value

    #         SR_approx_nxt_Bob = self.gamma_utils_cls.SR_active_concave_gamma_Bob(gamma_sol, gamma_0, p, cvx_bool=1)
    #         SR_approx_nxt_Eve = self.gamma_utils_cls.SR_active_concave_gamma_Eve(gamma_sol, gamma_0, p, cvx_bool=1)
    #         SSR_approx_nxt = max(SR_approx_nxt_Bob.value + SR_approx_nxt_Eve.value, 0)
    #         SEE_approx_nxt = SSR_approx_nxt / (ris_bool * np.real(np.trace(R @ (gamma_sol @ gamma_sol.conj().T))) + Pc_eq)

    #         status = problem.status
    #         solver_time = problem.solver_stats.solve_time

    #         print(f'gamma_opt step: {iter}, gamma_nxt_norm: {np.linalg.norm(gamma_sol)}, SSR_approx_nxt: {SSR_approx_nxt}, SEE_approx_nxt: {self.BW * SEE_approx_nxt}, Solver time: {solver_time}, Solver status: {status}')

    #         # if opt_bool == 0:
    #         #     ln_nxt = SSR_approx_nxt
    #         # else:
    #         #     ln_nxt = SEE_approx_nxt

    #         ln_nxt = SSR_approx_nxt if opt_bool == 0 else SEE_approx_nxt
        
    #     elapsed_time = time.time() - start_time
    #     tcpx_gamma = elapsed_time

    #     SSR_approx_sol = SSR_approx_nxt
    #     SEE_approx_sol = self.BW * SEE_approx_nxt

    #     if ln_nxt < ln_prev:
    #         print('\nStopping Optimization of gamma... optimal point reached!')
    #         gamma_sol = gamma_0
    #         SSR_approx_sol = SSR_approx_prev
    #         SEE_approx_sol = self.BW * SEE_approx_prev

    #     print(f'\ngamma_opt final -> Number of Steps: {iter}, SSR_opt: {SSR_approx_sol}, SEE_opt: {SEE_approx_sol}, time_complexity_gamma: {tcpx_gamma}, Solver status: {status}\n')
    #     return gamma_sol, SSR_approx_sol, SEE_approx_sol, iter, tcpx_gamma
    
    def gamma_cvxopt_algo1(self,
                                gamma_init: np.ndarray,
                                p: np.ndarray,
                                opt_bool: bool,
                                rf_state: str,
                                ris_state: str,
                                cons_state: str,
                                max_iters: int = 20,
                                tol: float = 1e-3
                                ) -> Tuple[np.ndarray, float, float, int, float]:
        """
        Algorithm 1: Iterative CVX-based gamma optimization with SCA.

        Uses successive concave approximations to optimize SSR or SEE.
        """
        
        # Initialize
        N = gamma_init.shape[0]
        gamma_prev = gamma_init.copy()
        ris_active = 1 if ris_state == 'active' else 0 # (ris_state == 'active')
        R = self.utils_cls.compute_R(self.H, p, self.sigma_RIS_sq)
        Rnorm = R / np.linalg.norm(R, 'fro')
        Pc_eq = self.gamma_utils_cls.compute_Pc_eq_gamma(p)
        PRmax_val = 0.0
        if ris_active:
            PRmax_val = ((self.a - 1) * np.real(np.trace(Rnorm))
                         if rf_state == 'RF-Gain' else self.PRmax)

        # Initial objective value
        SR_B0 = self.gamma_utils_cls.SR_active_concave_gamma_Bob(
            gamma_prev, gamma_prev, p, cvx_bool=1)
        SR_E0 = self.gamma_utils_cls.SR_active_concave_gamma_Eve(
            gamma_prev, gamma_prev, p, cvx_bool=1)
        SSR_prev = max(SR_B0.value + SR_E0.value, 0)
        SEE_prev = SSR_prev / (ris_active * np.real(np.trace(R @ (gamma_prev @ gamma_prev.conj().T)) ) + Pc_eq)
        ln_prev = SSR_prev if not opt_bool else SEE_prev

        print(f"[gamma_opt_algo1] Start optimization: SSR={SSR_prev:.6f}, SEE={SEE_prev:.6f}, opt_bool={opt_bool}")
        start_all = time.perf_counter()

        # Iterative loop
        for it in range(1, max_iters + 1):
            print(f"[gamma1] Iter {it}")
            # CVX variable
            gamma_var = cp.Variable((N,1), complex=True)
            # Build concave approximations
            SR_B = self.gamma_utils_cls.SR_active_concave_gamma_Bob(
                gamma_var, gamma_prev, p, cvx_bool=0)
            SR_E = self.gamma_utils_cls.SR_active_concave_gamma_Eve(
                gamma_var, gamma_prev, p, cvx_bool=0)
            obj = SR_B + SR_E
            # SEE adjustment
            if opt_bool and ris_active:
                obj -= ln_prev * ris_active * cp.real(cp.sum(cp.quad_form(gamma_var, R))) + ln_prev * Pc_eq

            # Constraints
            cons = []
            if ris_active:
                # reflection constraint linearized
                cons.append(
                    cp.real(gamma_prev.conj().T @ Rnorm @ gamma_prev)
                    + 2*cp.real(gamma_prev.conj().T @ Rnorm @ (gamma_var - gamma_prev))
                    >= np.real(np.trace(Rnorm))
                )
                if rf_state == 'RF-Power':
                    cons.append(
                        cp.real(cp.quad_form(gamma_var, R)) <= PRmax_val + np.real(np.trace(R))
                    )
                else:
                    cons.append(
                        cp.real(cp.quad_form(gamma_var, Rnorm)) <= PRmax_val + np.real(np.trace(Rnorm))
                    )
            else:
                if cons_state == 'global':
                    cons.append(
                        cp.real(cp.quad_form(gamma_var, Rnorm)) <= np.real(np.trace(Rnorm))
                    )
                else:
                    cons.append(cp.abs(gamma_var) <= 1)

            # Solve
            prob = cp.Problem(cp.Maximize(obj), cons)
            t0 = time.perf_counter()
            try:
                prob.solve(solver='MOSEK', warm_start=True)
            except cp.error.SolverError:
                print("MOSEK failed, switching to SCS...")
                prob.solve(solver='SCS', warm_start=True)
            t1 = time.perf_counter()
            solver_time = t1 - t0
            status = prob.status
            gamma_curr = gamma_var.value if (prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and gamma_var.value is not None) else gamma_prev

            # Compute approximations at new gamma
            SR_Bn = self.gamma_utils_cls.SR_active_concave_gamma_Bob(
                gamma_curr, gamma_curr, p, cvx_bool=1)
            SR_En = self.gamma_utils_cls.SR_active_concave_gamma_Eve(
                gamma_curr, gamma_curr, p, cvx_bool=1)
            SSR_curr = max(SR_Bn.value + SR_En.value, 0)
            SEE_curr = SSR_curr / (ris_active * np.real(np.trace(R @ (gamma_curr @ gamma_curr.conj().T))) + Pc_eq)
            ln_curr = SSR_curr if not opt_bool else SEE_curr

            print(f"[gamma_opt_algo1] it={it}, ||Δγ||={(np.linalg.norm(gamma_curr-gamma_prev)):.3e}, "
                  f"SSR={SSR_curr:.6f}, SEE={SEE_curr:.6f}, time={solver_time:.3f}s, status={status}")

            # Convergence
            if abs(ln_curr - ln_prev) < tol and status.startswith('optimal'):
                print("[gamma_opt_algo1] Converged within tolerance.")
                # Stopping: objective decrease
                if ln_curr < ln_prev:
                    print("[gamma_opt_algo1] Obj decreased; reverting to previous gamma.")
                    gamma_opt = gamma_prev
                    # SSR_opt, SEE_opt = SSR_prev, SEE_prev
                    break
                gamma_opt = gamma_curr
                # SSR_opt, SEE_opt = SSR_curr, SEE_curr
                break

            # prepare next iter
            gamma_opt = gamma_curr.copy()
            gamma_prev = gamma_curr.copy()
            SSR_prev, SEE_prev, ln_prev = SSR_curr, SEE_curr, ln_curr
        
        # Stopping: objective decrease
        # if ln_curr < ln_prev:
        #     print("[gamma_opt_algo1] Obj decreased; reverting to previous gamma.")
        #     gamma_opt = gamma_prev
        #     SSR_opt, SEE_opt = SSR_prev, SEE_prev
        # else:
        #     # max iters reached
        #     gamma_opt, SSR_opt, SEE_opt = gamma_curr, SSR_curr, SEE_curr
        
        total_time = time.perf_counter() - start_all
        # Compute approximations at new gamma
        SR_Bn = self.gamma_utils_cls.SR_active_concave_gamma_Bob(
            gamma_opt, gamma_opt, p, cvx_bool=1)
        SR_En = self.gamma_utils_cls.SR_active_concave_gamma_Eve(
            gamma_opt, gamma_opt, p, cvx_bool=1)
        SSR_final = max(SR_Bn.value + SR_En.value, 0)
        SEE_final = SSR_curr / (ris_active * np.real(np.trace(R @ (gamma_opt @ gamma_opt.conj().T))) + Pc_eq)
        
        print(f"[gamma_opt_algo1] Done. iters={it} -- SSR_opt={SSR_final:.6f}, SEE_opt={SEE_final:.6f}, total_time={total_time:.3f}s, status={status}")
        return gamma_opt, SSR_final, SEE_final, it, total_time
    
    

    def gamma_cvxopt_algo2(self,
                        gamma_init: np.ndarray,
                        p: np.ndarray,
                        opt_bool: bool,
                        rf_state: str,
                        ris_state: str,
                        cons_state: str,
                        extraction: str = 'pca',
                        num_randomizations: int = 50,
                        max_sca_iters: int = 20,
                        tol: float = 1e-3
                    ) -> Tuple[np.ndarray, float, float, int, float]:
        """
        Algorithm 2: SDR + SCA (+ Dinkelbach for SEE) gamma optimization.
        Returns:
        - gamma_opt (final beamforming vector),
        - SSR_final,
        - SEE_final,
        - number of SCA iterations used,
        - total solver time.
        """
        
         
        ris_active = 1 if ris_state == 'active' else 0
        Pc_eq = self.gamma_utils_cls.compute_Pc_eq_gamma(p)
        
        # Use, e.g., a 1e-6 slack:
        eps = 1e-6
        N = self.H.shape[0]
        R = self.utils_cls.compute_R(self.H, p, self.sigma_RIS_sq)
        Rnorm = R / np.linalg.norm(R, 'fro')
        
        PRmax = (self.a - 1) * np.real(np.trace(R)) if rf_state == 'RF-Gain' else self.PRmax
        
        # --- Precompute M once, before the SCA/Dinkelbach loop ---
        eigs = np.linalg.eigvalsh(R)
        pos = eigs[eigs > 0]
        if len(pos) == 0:
            # R is zero? then pick some large M like M = 1e3
            δ = 1e-6
            R += δ * np.eye(N)
            Rnorm = R / np.linalg.norm(R, 'fro')
            
            eigs = np.linalg.eigvalsh(R)
            pos = eigs[eigs > 0]
            
            lambda_min = float(np.min(pos))
            M = ( np.real(np.trace(R)) + PRmax ) / lambda_min
          
        else:
                  
            lambda_min = float(np.min(pos))
            M = ( np.real(np.trace(R)) + PRmax ) / lambda_min
            
        
        PRmax_val = 0.0
        if ris_active:
            if  rf_state == 'RF-Power':
                PRmax_val = (
                ((self.a - 1) * np.real(np.trace(R)))
                if rf_state == 'RF-Gain'
                else self.PRmax
            )
            else:
                PRmax_val = (
                    ((self.a - 1) * np.real(np.trace(Rnorm)))
                    if rf_state == 'RF-Gain'
                    else self.PRmax
                )
                       
        # --- Initialization (same as before) ---
        X0 = np.outer(gamma_init, gamma_init.conj())
        
        feasible, details = Utils.check_X_feasibility(
                    X0,
                    R,
                    Rnorm,
                    ris_state=ris_state,
                    rf_state=rf_state,
                    cons_state=cons_state,
                    PRmax_val=PRmax_val,
                    tol=1e-6
                )
        if not feasible:
            raise RuntimeError("Initial X0 is infeasible! Details:\n" + str(details))

        SSR_prev = (
            self.gamma_utils_cls.SR_concave_gamma_Bob_X(X0, p, X0=X0, cvx_bool=False)
            - self.gamma_utils_cls.SR_concave_gamma_Eve_X(X0, p, X0=X0, cvx_bool=False)
        )
        power_val = ris_active * np.real(np.trace(R @ X0)) + Pc_eq
        lam = SSR_prev / power_val if (power_val > 0) else 0.0
        total_solve_time = 0.0

        print("[gamma_opt_algo2] Starting SCA/Dinkelbach with max iters =", max_sca_iters)
        outer_start = time.time()

        for it in range(max_sca_iters):
            print(f"\n[gamma_opt_algo2] Iteration {it+1}/{max_sca_iters}")
            print(f" -- Previous SSR = {SSR_prev:.6e}, lambda = {lam:.6e}")

            # Setup SD variable and constraints
            X = cp.Variable((N, N), complex=True)
            constr = [X >> 0]
            constr += [X == X.H]
         
            # # 1) Define X as a real‐Hermitian PSD variable (no need for X == X.H).
            # X = cp.Variable((N, N), hermitian=True)
            # constr = [X >> 0]  # enforces X is Hermitian‐PSD

            # 2) Add the same “power” or “diagonal” constraints as before, but referencing X directly.
            if ris_active:
                if rf_state == 'RF-Power':
                    constr += [
                        cp.real(cp.trace(R @ X)) >= np.real(np.trace(R)), #- eps,
                        cp.real(cp.trace(R @ X)) <= PRmax_val + np.real(np.trace(R)), # + eps,
                        cp.real(cp.trace(X)) <= M,
                    ]
                     
                else:
                    # constr += [
                    #     cp.real(cp.trace(R @ X)) >= np.real(np.trace(R)) - eps,
                    #     cp.real(cp.trace(R @ X)) <= PRmax + np.real(np.trace(R)) + eps,
                    # ]
                    constr += [
                        cp.real(cp.trace(Rnorm @ X)) >= np.real(np.trace(Rnorm)), # - eps ,
                        cp.real(cp.trace(Rnorm @ X)) <= PRmax_val + np.real(np.trace(Rnorm)), # + eps,
                        cp.real(cp.trace(X)) <= M,
                    ]
            else:
                if cons_state == 'global':
                    constr += [cp.real(cp.trace(Rnorm @ X)) <= np.real(np.trace(Rnorm)), # + eps,
                               cp.real(cp.trace(X)) <= M,    
                               ]
                else:
                    constr += [cp.diag(X) <= 1,
                               cp.real(cp.trace(X)) <= float(N),                 
                               ]

            # 3) Build the concave surrogate for Bob and Eve exactly as before.
            Rb_surr = self.gamma_utils_cls.SR_concave_gamma_Bob_X(X, p, X0=X0, cvx_bool=True)
            Re_surr = self.gamma_utils_cls.SR_concave_gamma_Eve_X(X, p, X0=X0, cvx_bool=True)
            SSR_surr = Rb_surr - Re_surr
            
            # Don’t allow surrogate SSR < 0:
            constr += [ SSR_surr >= 0 ]

            if not opt_bool:
                objective = SSR_surr
            else:
                power_expr = ris_active * cp.real(cp.trace(R @ X)) + Pc_eq
                objective = SSR_surr - lam * power_expr

            # objective = 0 # feasibility check
            # objective = cp.real(cp.trace(X)) # or cp.real(cp.sum(cp.diag(X))) dummy check 
            # objective = Rb_surr # Test only Bob’s concave log-det part
            # objective = - Re_surr # Test only Eve’s concave log part
            # objective = SSR_surr # Test only Bob’s concave log-det part
            
            # 4) Solve with MOSEK.  We do NOT use warm_start=True here.
            prob = cp.Problem(cp.Maximize(objective), constr)
            t0 = time.perf_counter()
            try:
                # You can tweak these MOSEK parameters if you want tighter tolerances or more logging.
                mosek_params = {
                    # e.g. reduce feasibility tolerance (default ~1e-8)
                    "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-8,
                    "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-8,
                    # Relative gap tolerance (default ~1e-8).
                    # Stopping criterion based on (duality gap)/(1 + |objective|).
                    "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-8,
                    # turn off MOSEK output, or set to > 0 if you want logs.
                    "MSK_IPAR_LOG": 1,
                }
                # mosek_params["MSK_DPAR_INTPNT_TOL_REL_GAP"] = 1e-6
                # mosek_params["MSK_DPAR_INTPNT_TOL_PFEAS"]   = 1e-6
                # mosek_params["MSK_DPAR_INTPNT_TOL_DFEAS"]   = 1e-6
                
                prob.solve(
                    solver=cp.MOSEK,
                    mosek_params=mosek_params,
                    verbose=True,     # set True if you want MOSEK prints
                )
            except cp.error.SolverError as e:
                # If MOSEK absolutely fails, fall back to SCS just once.
                print(">>> MOSEK raised SolverError; falling back to SCS.  Error:", e)
                prob.solve(solver=cp.SCS, verbose=False)

            t1 = time.perf_counter()
            solver_time = t1 - t0
            total_solve_time += solver_time
            status = prob.status

            # 5) Retrieve X_opt, but if MOSEK is infeasible or inaccurate, fall back to X0.
            X_opt = X.value if (status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and X.value is not None) else X0
            feasible, details = Utils.check_X_feasibility(
                        X_opt,
                        R,
                        Rnorm,
                        ris_state=ris_state,
                        rf_state=rf_state,
                        cons_state=cons_state,
                        PRmax_val=PRmax_val,
                        tol=1e-6
                    )
            if not feasible:
                print("Warning: solver returned an infeasible X. Details:", details)   

            # 6) Compute true SSR and SEE at X_opt (same as before).
            SSR_curr = (
                self.gamma_utils_cls.SR_concave_gamma_Bob_X(X_opt, p, X0=X0, cvx_bool=False)
                - self.gamma_utils_cls.SR_concave_gamma_Eve_X(X_opt, p, X0=X0, cvx_bool=False)
            )
            power_val = ris_active * np.real(np.trace(R @ X_opt)) + Pc_eq
            SEE_curr = SSR_curr / power_val if (power_val > 0) else 0.0

            print(
                f"[gamma_opt_algo2] it={it} →  ||ΔX|| = {np.linalg.norm(X_opt - X0, 'fro'):.3e}, "
                f"SSR_curr = {SSR_curr:.6e}, SEE_curr = {SEE_curr:.6e}, solver_time = {solver_time:.4f}s, status = {status}"
            )

            # 7) Convergence test.
            if not opt_bool:
                if abs(SSR_curr - SSR_prev) < tol:
                    print(f" -- Converged (|ΔSSR| < {tol}).")
                       # 8) Safeguard: if SSR dropped, revert.
                    if it > 0 and (SSR_curr < SSR_prev or SSR_curr <= 0):
                        print(" -- SSR decreased or its has a negative Secrecy; reverting to X0.")
                        X_opt = X0
                        SSR_curr = SSR_prev
                        break
                    SSR_prev = SSR_curr
                    break
                
                SSR_prev = SSR_curr
            else:
                lam_new = SSR_curr / power_val if (power_val > 0) else 0.0
                print(f" -- Dinkelbach update: λ_new = {lam_new:.6e}")
                if abs(lam_new - lam) < tol:
                    print(f" -- Converged (|Δλ| < {tol}).")
                    # 8) Safeguard: if SSR dropped, revert.
                    if it > 0 and (lam_new < lam or lam_new <= 0):
                        print(" -- SEE decreased or its has a negative Secrecy; reverting to X0.")
                        X_opt = X0
                        lam_new = lam
                        break
                    lam = lam_new
                    break
                
                lam = lam_new

            # 9) Update X0 for next iteration
            X0 = X_opt

        # end of SCA / Dinkelbach loop
        total_solve_time = time.time() - outer_start

        # 10) Final gamma extraction (PCA or randomization) – unchanged
        print("\n[gamma_opt_algo2] Extracting gamma via", extraction)
        if extraction == "pca":
            eigvals, eigvecs = np.linalg.eigh(X_opt)
            principal = eigvecs[:, -1]
            gamma_opt = principal * np.sqrt(max(eigvals[-1], 0.0))
            gamma_opt = gamma_opt.reshape(-1, 1)
        else:
            best_val = -np.inf
            gamma_opt = None
            sqrtX = np.linalg.cholesky(X_opt + 1e-9 * np.eye(N))
            for _ in range(num_randomizations):
                z = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
                cand = sqrtX @ z
                if ris_active and rf_state == "RF-Power":
                    scale = np.sqrt((self.PRmax + np.trace(R)) / np.real(cand.conj().T @ R @ cand))
                    cand *= scale
                elif not ris_active and cons_state == "global":
                    Rnorm = R / np.linalg.norm(R, "fro")
                    scale = np.sqrt(np.trace(Rnorm) / np.real(cand.conj().T @ Rnorm @ cand))
                    cand *= scale

                SSR_val = (
                    self.gamma_utils_cls.SR_concave_gamma_Bob_X(
                        np.outer(cand, cand.conj()), p, X0=np.outer(cand, cand.conj()), cvx_bool=False
                    )
                    - self.gamma_utils_cls.SR_concave_gamma_Eve_X(
                        np.outer(cand, cand.conj()), p, X0=np.outer(cand, cand.conj()), cvx_bool=False
                    )
                )
                cand_power = ris_active * np.real(cand.conj().T @ R @ cand) + Pc_eq
                SEE_val = SSR_val / cand_power if (cand_power > 0) else 0.0
                val = SSR_val - (lam * cand_power if opt_bool else 0)
                if val > best_val:
                    best_val = val
                    gamma_opt = cand
            print(f" -- Randomization best SSR = {SSR_val:.6e}, SEE = {SEE_val:.6e}")

        # 11) Compute final metrics
        X_sol = gamma_opt @ gamma_opt.conj().T
        SSR_final = float(
            self.gamma_utils_cls.SR_concave_gamma_Bob_X(X_sol, p, X0=X_sol, cvx_bool=False)
            - self.gamma_utils_cls.SR_concave_gamma_Eve_X(X_sol, p, X0=X_sol, cvx_bool=False)
        )
        power_final = ris_active * np.real(gamma_opt.conj().T @ R @ gamma_opt) + Pc_eq
        SEE_final = float(SSR_final / power_final if (power_final > 0) else 0.0)

        print(
            f"\n[gamma_opt_algo2] Done (it={it}). SSR_final = {SSR_final:.6e}, "
            f"SEE_final = {SEE_final:.6e}, total_solve_time = {total_solve_time:.4f}s, status = {status}\n"
        )
        return gamma_opt, SSR_final, SEE_final, it, total_solve_time

    # def gamma_cvxopt_algo2(self,
    #                 gamma_init: np.ndarray,
    #                 p: np.ndarray,
    #                 opt_bool: bool,
    #                 rf_state: str,
    #                 ris_state: str,
    #                 cons_state: str,
    #                 extraction: str = 'pca',
    #                 num_randomizations: int = 50,
    #                 max_sca_iters: int = 20,
    #                 tol: float = 1e-3
    #             ) -> Tuple[np.ndarray, float, float, int, float]:
    #     """
    #     Algorithm 2: SDR + SCA (+ Dinkelbach for SEE) gamma optimization.

    #     Returns final gamma via PCA or randomization after convergence,
    #     with a safeguard to retain the previous solution if the surrogate
    #     objective decreases. Always returns SSR and SEE values.
    #     """

    #     # Problem dimensions and constants
    #     N = self.H.shape[0]
    #     R = self.utils_cls.compute_R(self.H, p, self.sigma_RIS_sq)
    #     Rnorm = R / np.linalg.norm(R, 'fro')
    #     ris_active = 1 if ris_state == 'active' else 0 # (ris_state == 'active')
    #     Pc_eq = self.gamma_utils_cls.compute_Pc_eq_gamma(p)
    #     PRmax_val = 0.0
    #     if ris_active:
    #         PRmax_val = ((self.a - 1) * np.real(np.trace(Rnorm))
    #                      if rf_state == 'RF-Gain' else self.PRmax)

    #     # Initialization
    #     X0 = np.outer(gamma_init, gamma_init.conj()) # np.eye(N)
    #     SSR_prev = (self.gamma_utils_cls.SR_concave_gamma_Bob_X(X0, p, X0=X0, cvx_bool=False)
    #                     - self.gamma_utils_cls.SR_concave_gamma_Eve_X(X0, p, X0=X0, cvx_bool=False))
    #     # SSR_prev = -np.inf
    #     power_val = ris_active * np.real(np.trace(R @ X0)) + Pc_eq
    #     lam = SSR_prev / power_val
    #     # lam = 0.0
    #     total_solve_time = 0.0

    #     print("[gamma_opt_algo2] Starting SCA (and Dinkelbach if SEE) with max iters=", max_sca_iters)
        
    #     start = time.time()
    #     # Outer SCA/Dinkelbach loop
    #     for it in range(max_sca_iters):
    #         print(f"\n[gamma_opt_algo2] Iteration {it+1}/{max_sca_iters}")
    #         print(f" -- Previous SSR: {SSR_prev:.6f}, lambda: {lam:.6f}")

    #         # Setup SD variable and constraints
    #         X = cp.Variable((N, N), complex=True)
    #         constr = [X >> 0]
    #         constr += [X == X.H]
    #         if ris_active:
    #             if rf_state == 'RF-Power':
    #                 constr += [cp.real(cp.trace(R @ X)) >=  np.real(np.trace(R))]
    #                 constr += [cp.real(cp.trace(R @ X)) <= PRmax_val +  np.real(np.trace(R))]
    #             else:         
    #                 constr += [cp.real(cp.trace(Rnorm @ X)) >= np.real(np.trace(Rnorm))]
    #                 constr += [cp.real(cp.trace(Rnorm @ X)) <= PRmax_val +  np.real(np.trace(Rnorm))]
    #         else:
    #             if cons_state == 'global':
    #                 constr += [cp.real(cp.trace(Rnorm @ X)) <= np.real(np.trace(Rnorm))]
    #             else:
    #                 constr += [cp.diag(X) <= 1]

    #         # Build concave surrogate objective
    #         Rb_surr = self.gamma_utils_cls.SR_concave_gamma_Bob_X(X, p, X0=X0, cvx_bool=True)
    #         Re_surr = self.gamma_utils_cls.SR_concave_gamma_Eve_X(X, p, X0=X0, cvx_bool=True)
    #         SSR_surr = Rb_surr - Re_surr

    #         if not opt_bool:
    #             objective = SSR_surr
    #         else:
    #             power_expr = ris_active * cp.real(cp.trace(R @ X)) + Pc_eq
    #             objective = SSR_surr - lam * power_expr

    #         # Solve SDP
    #         prob = cp.Problem(cp.Maximize(objective), constr) 
    #         t0 = time.perf_counter()
    #         try:
    #             prob.solve(solver='MOSEK', warm_start=True)
    #         except cp.error.SolverError:
    #             print("MOSEK failed, switching to SCS...")
    #             prob.solve(solver='SCS', warm_start=True)
    #         t1 = time.perf_counter()
    #         solver_time = t1 - t0
    #         status = prob.status
    #         # prob.solve(solver='MOSEK', warm_start=True)
    #         # duration = time.time() - start
    #         # total_solve_time += duration
           
    #         # print(f" -- Solver status: {status}, solve_duration: {duration:.4f}s")

    #         # Retrieve X_opt and compute true SSR
    #         X_opt = X.value if (prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and X.value is not None) else X0

    #         SSR_curr = (self.gamma_utils_cls.SR_concave_gamma_Bob_X(X_opt, p, X0=X0, cvx_bool=False)
    #                     - self.gamma_utils_cls.SR_concave_gamma_Eve_X(X_opt, p, X0=X0, cvx_bool=False))
    #         # print(f" -- SSR_curr: {SSR_curr:.6f}")

    #         # Dinkelbach preliminary step for SEE
    #         # Extract PCA for power calculation
    #         # eigvals, eigvecs = np.linalg.eigh(X_opt)
    #         # principal = eigvecs[:, -1] * np.sqrt(max(eigvals[-1], 0.0))
    #         # power_val = ris_active * np.real(principal.conj().T @ R @ principal) + Pc_eq
    #         power_val = ris_active * np.real(np.trace(R @ X_opt)) + Pc_eq
    #         SEE_curr = SSR_curr / power_val
    #         print(f"[gamma_opt_algo2] it={it} -- ||ΔX||={(np.linalg.norm(X_opt - X0, 'fro')):.3e}, SSR_curr: {SSR_curr:.6f}, SEE_curr: {SEE_curr:.6f}, solve_duration: {solver_time:.4f}s,  Solver status: {status}")

    #         # Safeguard: if SSR decreased, retain previous solution
    #         if it > 0 and SSR_curr < SSR_prev:
    #             print(" -- SSR decreased, reverting to previous solution.")
    #             SSR_curr = SSR_prev
    #             X_opt = X0
    #             break
                    
    #         # Check convergence
    #         if not opt_bool:
    #             if abs(SSR_curr - SSR_prev) < tol:
    #                 print(f" -- Converged SSR change < tol ({tol}).")
    #                 SSR_prev = SSR_curr
    #                 break
    #             SSR_prev = SSR_curr
    #         else:
    #             lam_new = SSR_curr / power_val if power_val > 0 else 0.0
    #             print(f" -- Dinkelbach update lambda_new: {lam_new:.6f}")
    #             if abs(lam_new - lam) < tol:
    #                 print(f" -- Converged lambda change < tol ({tol}).")
    #                 lam = lam_new
    #                 break
    #             lam = lam_new

    #         # Update X0
    #         X0 = X_opt

    #     total_solve_time = time.time() - start
    #     # Final gamma extraction
    #     print("\n[gamma_opt_algo2] Extracting gamma via", extraction)
    #     if extraction == 'pca':
    #         eigvals, eigvecs = np.linalg.eigh(X_opt)
    #         principal = eigvecs[:, -1]
    #         gamma_opt = principal * np.sqrt(max(eigvals[-1], 0.0))
    #         gamma_opt = gamma_opt[:, None] # gamma_opt.reshape(-1, 1)
    #     else:
    #         best_val = -np.inf
    #         gamma_opt = None
    #         sqrtX = np.linalg.cholesky(X_opt + 1e-9 * np.eye(N))
    #         for _ in range(num_randomizations):
    #             z = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
    #             cand = sqrtX @ z
    #             if ris_active and rf_state=='RF-Power':
    #                 scale = np.sqrt((self.PRmax + np.trace(R)) / np.real(cand.conj().T @ R @ cand))
    #                 cand *= scale
    #             elif not ris_active and cons_state=='global':
    #                 Rnorm = R / np.linalg.norm(R, 'fro')
    #                 scale = np.sqrt(np.trace(Rnorm) / np.real(cand.conj().T @ Rnorm @ cand))
    #                 cand *= scale
    #             SSR_val = (self.gamma_utils_cls.SR_concave_gamma_Bob_X(np.outer(cand, cand.conj()), p, X0=np.outer(cand, cand.conj()), cvx_bool=False)
    #                     - self.gamma_utils_cls.SR_concave_gamma_Eve_X(np.outer(cand, cand.conj()), p, X0=np.outer(cand, cand.conj()), cvx_bool=False))
    #             cand_power = ris_active * np.real(cand.conj().T @ R @ cand) + Pc_eq
    #             SEE_val = SSR_val / cand_power
    #             val = SSR_val - (lam * cand_power if opt_bool else 0)
    #             if val > best_val:
    #                 best_val, gamma_opt = val, cand
    #         print(f" -- Randomization best SSR: {SSR_val:.6f}, SEE: {SEE_val:.6f}")

        
    #     # Final metrics
    #     X_sol = gamma_opt @ gamma_opt.conj().T
    #     SSR = (self.gamma_utils_cls.SR_concave_gamma_Bob_X(X_sol, p, X0=X_sol, cvx_bool=False)
    #                     - self.gamma_utils_cls.SR_concave_gamma_Eve_X(X_sol, p, X0=X_sol, cvx_bool=False))
    #     SSR_final = float(SSR)
    #     SEE_final = float(SSR_final / (ris_active * np.real(gamma_opt.conj().T @ R @ gamma_opt) + Pc_eq))
    #     print(f"\n[gamma_opt_algo2] Done. it={it} --  SSR_final: {SSR_final:.6f}, SEE_final: {SEE_final:.6f}, total_solve_time: {total_solve_time:.4f}s, status: {status}\n")
    #     return gamma_opt, SSR_final, SEE_final, it, total_solve_time


    def optimize_gamma_ls(self,
                        gamma_init: np.ndarray,
                        p: np.ndarray,
                        opt_ee: bool,
                        rf_state: str,
                        ris_state: str,
                        cons_state: str,
                        tol: float = 1e-3,
                        max_cycles: int = 20
                        ) -> Tuple[np.ndarray, float, float, int, float]:
        """
        Coordinate‐descent + 1D line‐search for optimizing gamma, with:
        - Final per‐element amplitude clamping based on (ris_state, cons_state).
        - Optional global‐norm check if cons_state=='global'.

        Parameters
        ----------
        gamma_init : (N,) complex
            Initial RIS coefficients.
        p : (K,) float
            UE powers.
        opt_ee : bool
            If False, maximize SSR. If True, maximize SEE.
        rf_state : {'RF-Gain','RF-Power'}
        ris_state : {'active','passive'}
        cons_state : {'global','local'}
        tol : float
            Convergence tolerance for objective change.
        max_cycles : int
            Maximum alternating cycles (amplitude+phase updates).

        Returns
        -------
        gamma_opt : (N,) complex
            Final RIS coefficients (clamped/scaled as needed).
        SSR_opt : float
            Secrecy sum‐rate at gamma_opt.
        SEE_opt : float
            Secrecy energy efficiency at gamma_opt (if opt_ee=True).
        cycles_run : int
            Number of coordinate‐descent cycles actually run.
        elapsed : float
            Wall‐clock time spent in this routine (seconds).
        """
        N = gamma_init.size
        # Separate amplitude and phase
        a_vec = np.abs(gamma_init).copy()
        phi_vec = np.mod(np.angle(gamma_init), 2 * np.pi).copy()
        
        # Precompute R and its trace
        R = self.utils_cls.compute_R(self.H, p, self.sigma_RIS_sq)  # (N×N) Hermitian
        trace_R = np.real(np.trace(R))
        ris_bool = 1 if ris_state == 'active' else 0

        # Determine PRmax (0 if passive, else as given)
        if ris_state == 'active':
            if rf_state == 'RF-Gain':
                PRmax = (self.a - 1) * trace_R
            else:
                PRmax = self.PRmax
        else:
            PRmax = 0.0

        # Numeric-only objective: SSR or SSR/(Pris+Pc)
        def objective(a: np.ndarray, phi: np.ndarray) -> float:
            gamma_vec = a * np.exp(1j * phi)
            # Bob’s sum-rate
            SRb = float(
                self.utils_cls.SR_active_algo1(
                    self.G_B, self.H, gamma_vec, p,
                    self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq,
                    self.scsi_bool, True, "Bob"
                )
            )
            # Eve’s approx rate
            if self.scsi_bool == 0:
                
                SRe = float(
                self.utils_cls.SR_active_algo1(
                    self.G_E, self.H, gamma_vec, p,
                    self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq,
                    self.scsi_bool, True, "Eve"
                )
            )
            else:
                SRe = float(
                self.utils_cls.SR_active_algo1(
                    self.G_E, self.H, gamma_vec, p,
                    self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq,
                    self.scsi_bool, False, "Eve"
                )
            )
                
            
            SSR = max(SRb - SRe, 0.0)

            if not opt_ee:
                return SSR

            # Otherwise: SEE = SSR / (Pris + Pc_eq)
            Pc_eq = self.gamma_utils_cls.compute_Pc_eq_gamma(p)
            Pris = ris_bool * np.real(np.vdot(gamma_vec, R @ gamma_vec))
            return SSR / (Pris + Pc_eq + 1e-16)

        prev_obj = objective(a_vec, phi_vec) #-np.inf
        gamma_prev = a_vec * np.exp(1j * phi_vec)
        cycles_run = 0
        start_time = time.perf_counter()

        

        print("\n[gamma_ls] Starting coordinate‐descent line‐search")
        for cycles in range(1, max_cycles + 1):
            cycles_run = cycles
            gamma_curr = a_vec * np.exp(1j * phi_vec)
            curr_obj = objective(a_vec, phi_vec)
            print(f" Cycle {cycles:2d}: current objective = {curr_obj:.6e}")

            # Amplitude updates
            for n in range(N):
                gamma_full = a_vec * np.exp(1j * phi_vec)
                gamma_nn_sq = np.abs(gamma_full[n])**2
                Rnn = np.real(R[n, n])

                quad_full = np.real(np.vdot(gamma_full, R @ gamma_full))
                quad_excl_n = quad_full - (gamma_nn_sq * Rnn)

                # Determine per-element [a_lo, a_hi]
                if cons_state == 'global':
                    # Global constraint (PRmax = 0 if passive)
                    lo_val = (trace_R - quad_excl_n) / Rnn
                    a_lo = float(np.sqrt(lo_val)) if lo_val > 0 else 0.0

                    hi_val = (trace_R + PRmax - quad_excl_n) / Rnn
                    a_hi = float(np.sqrt(hi_val)) if hi_val > 0 else 1.0

                    # If active + local would not happen here—global branch only
                else:
                    # cons_state == 'local'
                    if ris_state == 'active':
                        # Active + Local: [1, a_max] with a_max from PRmax
                        a_lo = 1.0
                        hi_val = (trace_R + PRmax - quad_excl_n) / Rnn
                        a_hi = float(np.sqrt(hi_val)) if hi_val > 0 else 1.0
                    else:
                        # Passive + Local: [0, 1]
                        a_lo = 0.0
                        a_hi = 1.0

                # Final sanity clamp
                if a_lo < 0.0:
                    a_lo = 0.0
                if a_hi <= 0.0:
                    a_hi = 1.0
                if a_hi < a_lo:
                    temp = a_hi
                    a_hi = a_lo
                    a_lo = temp

                def f_a(x: float) -> float:
                    temp_a = a_vec.copy()
                    temp_a[n] = x
                    return objective(temp_a, phi_vec)

                best_a = _line_search(f_a, a_lo, a_hi, tol)
                a_vec[n] = best_a
                print(f"   [amp n={n:2d}] a_lo={a_lo:.4f}, a_hi={a_hi:.4f} → a[{n}]={best_a:.4f}")

            # Phase updates
            for n in range(N):
                def f_phi(x: float) -> float:
                    temp_phi = phi_vec.copy()
                    temp_phi[n] = x
                    return objective(a_vec, temp_phi)

                best_phi = _line_search(f_phi, 0.0, 2*np.pi, tol)
                phi_vec[n] = best_phi
                print(f"   [phase n={n:2d}] phi[{n}]={best_phi:.4e}")

            new_obj = objective(a_vec, phi_vec)
            print(f" Cycle {cycles:2d} done: new objective = {new_obj:.6e}")

            if abs(new_obj - curr_obj) < tol:
                print(" [gamma_ls] Converged (objective change below tol).")
                if new_obj < curr_obj:
                    print(" [gamma_ls] Objective decreased; reverting to previous gamma.")
                    gamma_opt = gamma_prev
                else:
                    gamma_opt = a_vec * np.exp(1j * phi_vec)
                break

            prev_obj = curr_obj
            gamma_prev = gamma_curr.copy()
        else:
            gamma_opt = a_vec * np.exp(1j * phi_vec)
            print(" [gamma_ls] Reached max_cycles without full convergence.")

        # if cycles_run == max_cycles + 1:
        #     gamma_opt = a_vec * np.exp(1j * phi_vec)
        #     print(" [gamma_ls] Reached max_cycles without full convergence.")
        
        # ────────────────────────────────────────────────────────────────────────────
        # 1) FINAL PER‐ELEMENT AMPLITUDE CLAMPING, same logic as above
        gamma_opt = np.array(gamma_opt, copy=True)
        a_opt = np.abs(gamma_opt)
        phi_opt = np.mod(np.angle(gamma_opt), 2 * np.pi) #  np.angle(gamma_opt)

        for n in range(N):
            gamma_full = a_opt * np.exp(1j * phi_opt)
            gamma_nn_sq = a_opt[n]**2
            Rnn = np.real(R[n, n])
            quad_full = np.real(np.vdot(gamma_full, R @ gamma_full))
            quad_excl_n = quad_full - (gamma_nn_sq * Rnn)

            # Per-element [a_lo, a_hi]
            if cons_state == 'global':
                lo_val = (trace_R - quad_excl_n) / Rnn
                a_lo = float(np.sqrt(lo_val)) if lo_val > 0 else 0.0
                hi_val = (trace_R + PRmax - quad_excl_n) / Rnn
                a_hi = float(np.sqrt(hi_val)) if hi_val > 0 else 1.0
            else:
                if ris_state == 'active':
                    a_lo = 1.0
                    hi_val = (trace_R + PRmax - quad_excl_n) / Rnn
                    a_hi = float(np.sqrt(hi_val)) if hi_val > 0 else 1.0
                else:
                    a_lo = 0.0
                    a_hi = 1.0

            if a_opt[n] < a_lo:
                print(f" [gamma_ls] Clamping a[{n}] up from {float(a_opt[n]):.4f} to {a_lo:.4f}")
                a_opt[n] = a_lo
            elif a_opt[n] > a_hi:
                print(f" [gamma_ls] Clamping a[{n}] down from {float(a_opt[n]):.4f} to {a_hi:.4f}")
                a_opt[n] = a_hi

        gamma_opt = a_opt * np.exp(1j * phi_opt)

        # ────────────────────────────────────────────────────────────────────────────
        # 2) GLOBAL QUADRATIC CHECK (only if cons_state=='global')
        if cons_state == 'global':
            eigs = np.linalg.eigvalsh(R)
            pos_eigs = eigs[eigs > 0]
            if pos_eigs.size == 0:
                raise RuntimeError("All eigenvalues of R are non-positive; cannot enforce global constraint.")

            lambda_min = float(np.min(pos_eigs))
            lambda_max = float(np.max(pos_eigs))
            M_min = trace_R / lambda_max
            M_max = (trace_R + PRmax) / lambda_min

            norm_sq = np.linalg.norm(gamma_opt)**2
            if norm_sq < M_min - 1e-12:
                scale = np.sqrt(M_min / (norm_sq + 1e-16))
                print(f" [gamma_ls] Scaling γ up: ||γ||²={norm_sq:.4e} < M_min={M_min:.4e} → scale={scale:.4e}")
                gamma_opt *= scale
            elif norm_sq > M_max + 1e-12:
                scale = np.sqrt(M_max / norm_sq)
                print(f" [gamma_ls] Scaling γ down: ||γ||²={norm_sq:.4e} > M_max={M_max:.4e} → scale={scale:.4e}")
                gamma_opt *= scale
            else:
                print(f" [gamma_ls] ||γ||²={norm_sq:.4e} within [{M_min:.4e}, {M_max:.4e}]")
        else:
            print(f" [gamma_ls] Skipped global‐norm check (cons_state={cons_state}).")

        elapsed = time.perf_counter() - start_time

        GE_True = self.G_E if self.scsi_bool == 0 else  self.G_E + self.GE_error
        
        # Compute final SSR / SEE at gamma_opt
        SRb_final = float(
            self.utils_cls.SR_active_algo1(
                self.G_B, self.H, gamma_opt, p,
                self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq,
                self.scsi_bool, True, "Bob"
            )
        )
        SRe_final = float(
            self.utils_cls.SR_active_algo1(
                GE_True, self.H, gamma_opt, p,
                self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq,
                self.scsi_bool, True, "Eve"
            )
        )
        SSR_opt = max(SRb_final - SRe_final, 0.0)

        Pc_eq = self.gamma_utils_cls.compute_Pc_eq_gamma(p)
        Pris = ris_bool * np.real(np.vdot(gamma_opt, R @ gamma_opt))
        SEE_opt = SSR_opt / (Pris + Pc_eq + 1e-16)
        
        # if opt_ee:
        #     Pc_eq = self.gamma_utils_cls.compute_Pc_eq_gamma(p)
        #     Pris = ris_bool * np.real(np.vdot(gamma_opt, R @ gamma_opt))
        #     SEE_opt = SSR_opt / (Pris + Pc_eq + 1e-16)
        # else:
        #     SEE_opt = None

        print(f"\n[gamma_ls] final → cycles={cycles_run}, ||γ||={np.linalg.norm(gamma_opt):.3e}, "
            f"SSR={SSR_opt:.6e}, SEE={SEE_opt:.6e}, time={elapsed:.3f}s\n") # SEE={SEE_opt:.6e if SEE_opt is not None else 'N/A'}

        return gamma_opt, SSR_opt, SEE_opt, cycles_run, elapsed


    # def optimize_gamma_ls(self,
    #                       gamma_init: np.ndarray,
    #                       p: np.ndarray,
    #                       opt_ee: bool,
    #                       rf_state: str,
    #                       ris_state: str,
    #                       cons_state: str,
    #                       tol: float = 1e-3,
    #                       max_cycles: int = 20) -> Tuple[np.ndarray, float, float, int, float]:
    #     """Coordinate descent with 1D line search for gamma."""
    #     N = gamma_init.size
    #     a_vec = np.abs(gamma_init).copy()
    #     phi_vec = np.angle(gamma_init).copy()
    #     phi_vec = np.mod(phi_vec, 2*np.pi)
    #     prev_obj = -np.inf
    #     ris_bool = 1 if ris_state == 'active' else 0

    #     # Precompute R
    #     R = self.utils_cls.compute_R(self.H, p, self.sigma_RIS_sq)
    #     # Rnorm = R / np.linalg.norm(R, 'fro')
    #     trace_R = np.real(np.trace(R))
    #     if ris_state == 'active':
    #         PRmax = (self.a - 1)* np.real(np.trace(R)) if rf_state == 'RF-Gain' else self.PRmax
    #     else:
    #         PRmax = 0
            
    #     # eigs = np.linalg.eigvalsh(R)
    #     # Rmax, Rmin = eigs.max(), eigs.min()
    #     # a_lo_q = np.sqrt(trace_R / (Rmax * N)) if ris_state == 'active' else 0
    #     # a_hi_q = np.sqrt((trace_R + (PRmax if ris_state == 'active' else 0)) / (Rmin * N)) if cons_state == 'global' else 1
    #     cycles = 0
    #     start = time.time()

    #     def objective(a, phi):
    #         gamma = a * np.exp(1j * phi)
    #         SRb = self.gamma_utils_cls.SR_active_concave_gamma_Bob(gamma, gamma, p, cvx_bool=1).value
    #         SRe = self.gamma_utils_cls.SR_active_concave_gamma_Eve(gamma, gamma, p, cvx_bool=1).value
    #         SSR = max(SRb + SRe, 0)
    #         if not opt_ee:
    #             return SSR
    #         Pc_eq = self.gamma_utils_cls.compute_Pc_eq_gamma(p)
    #         # diag_R   = np.real(np.diag(R))        # [R₁₁, R₂₂, …, R_NN]
    #         Pris  = ris_bool * np.real(np.trace(R @ (gamma @ gamma.conj().T))) # np.dot(diag_R, a**2)
    #         # Pris = (trace_R + (self.PRmax if ris_state == 'active' else 0)) * np.real(np.dot(a**2, np.diag(R)/trace_R))
    #         return SSR / (Pris + Pc_eq)

    #     while cycles < max_cycles:
    #         cycles += 1
    #         # gamma = a_vec * np.exp(1j * phi_vec)
    #         # amplitude updates
    #         for n in range(N):
    #             gamma = a_vec * np.exp(1j * phi_vec)
    #             gamma_n_sq = np.abs(gamma[n])**2
    #             Rnn_real = np.real(R[n, n])

    #             # Compute full quadratic form
    #             quad_form = np.real(np.trace(R @ (gamma @ gamma.conj().T))) # np.real(np.vdot(gamma, R @ gamma))

    #             # Remove only the (n,n) term from the full quadratic form
    #             gamma_n_contrib = gamma_n_sq * Rnn_real
    #             quad_form_excl_n = quad_form - gamma_n_contrib
    #             lo = (trace_R - quad_form_excl_n) / Rnn_real
    #             a_lo = (np.sqrt(lo) if lo > 0 else 0 ) if ris_state == 'active' else 0
    #             a_hi = np.sqrt((PRmax + trace_R - quad_form_excl_n) / Rnn_real) if cons_state == 'global' else 1
    #             if a_hi <= 0:
    #                 a_hi = 1          
    #             def f_a(x):
    #                 a_temp = a_vec.copy()
    #                 a_temp[n] = x
    #                 return objective(a_temp, phi_vec)
    #             a_vec[n] = _line_search(f_a, a_lo, a_hi, tol)
    #         # phase updates
    #         for n in range(N):
    #             def f_phi(x):
    #                 phi_temp = phi_vec.copy()
    #                 phi_temp[n] = x
    #                 return objective(a_vec, phi_temp)
    #             phi_vec[n] = _line_search(f_phi, 0, 2*np.pi, tol)

    #         curr_obj = objective(a_vec, phi_vec)
    #         if abs(curr_obj - prev_obj) < tol:
    #             print('\nStopping Optimization of gamma... optimal point reached!')
    #             if curr_obj < prev_obj:
    #                 gamma_sol = gamma_prev
    #             else:
    #                 gamma_sol = a_vec * np.exp(1j * phi_vec)
    #             break
    #         prev_obj = curr_obj
    #         gamma_prev = gamma 
            
    #     gamma_opt = gamma_sol # a_vec * np.exp(1j * phi_vec)
    #     SSR_opt = objective(a_vec, phi_vec) if not opt_ee else None
    #     SEE_opt = objective(a_vec, phi_vec) if opt_ee else None
    #     elapsed = time.time() - start
    #     # ensure both SSR and SEE computed
    #     if SSR_opt is None:
    #         gamma = gamma_opt
    #         SRb = self.gamma_utils_cls.SR_active_concave_gamma_Bob(gamma, gamma, p, cvx_bool=1).value
    #         SRe = self.gamma_utils_cls.SR_active_concave_gamma_Eve(gamma, gamma, p, cvx_bool=1).value
    #         SSR_opt = max(SRb + SRe, 0)
    #     if SEE_opt is None:
    #         gamma = gamma_opt
    #         SRb = self.gamma_utils_cls.SR_active_concave_gamma_Bob(gamma, gamma, p, cvx_bool=1).value
    #         SRe = self.gamma_utils_cls.SR_active_concave_gamma_Eve(gamma, gamma, p, cvx_bool=1).value
    #         SSR_val = max(SRb + SRe, 0)
    #         Pc_eq = self.gamma_utils_cls.compute_Pc_eq_gamma(p)
    #         Pris  = ris_bool * np.real(np.trace(R @ (gamma @ gamma.conj().T)))
    #         # Pris = (trace_R + (self.PRmax if ris_state=='active' else 0)) * np.real(np.dot(a_vec**2, np.diag(R)/trace_R))
    #         SEE_opt = SSR_val / (Pris + Pc_eq)
        
    #     print(f'\n[gamma_ls] final -> Number of Steps: {cycles}, gamma_norm: {np.linalg.norm(gamma_opt):.3e}, SSR_opt: {SSR_opt:.6f}, SEE_opt: {SEE_opt:.6f}, time_complexity_gamma: {elapsed:.4f}\n')

    #     return gamma_opt, SSR_opt, SEE_opt, cycles, elapsed

    
    def p_cvxopt_algo1(self,
                       gamma: np.ndarray,
                       p_init: np.ndarray,
                       opt_bool: bool,
                       ris_state: str,
                       max_iters: int = 20,
                       tol: float = 1e-3
                       ) -> Tuple[np.ndarray, float, float, int, float]:
        """
        CVX-based power allocation (Algorithm 1) with successive concave approximation.

        Parameters
        ----------
        gamma : (N,) complex
            Current RIS reflection coefficients.
        p_init : (K,) float
            Initial UE power allocation.
        opt_bool : bool
            False -> maximize SSR; True -> maximize SEE.
        ris_state : {'active','passive'}
        max_iters : int
            Maximum alternating iterations.
        tol : float
            Convergence tolerance on objective.

        Returns
        -------
        p_sol : (K,) float
            Optimized power allocation.
        SSR_opt : float
            Final secrecy sum-rate.
        SEE_opt : float
            Final secrecy energy-efficiency.
        iters : int
            Number of iterations run.
        runtime : float
            Total optimization time in seconds.
        """
        print("[p_opt] Starting power optimization...")
        logger.info("Power optimization initiated.")
        start_all = time.perf_counter()

        # Initial values
        p_prev = p_init.copy()
        K = p_prev.size
        # Compute static terms
        Pc_eq = self.power_utils_cls.compute_Pc_eq_p(gamma)
        if ris_state == 'active':
            mu_eq = self.power_utils_cls.compute_mu_eq_p(gamma)
        else:
            mu_eq = np.full(K, self.mu)

        # Initial objective
        SSR_expr = self.power_utils_cls.SSR_active_concave_p(gamma, p_prev, p_prev, cvx_bool=1)
        SSR_prev = max(SSR_expr.value, 0)
        denom_prev = mu_eq @ p_prev + Pc_eq
        SEE_prev = SSR_prev / denom_prev
        ln_prev = SSR_prev if not opt_bool else SEE_prev
        print(f"[p_opt] init SSR={SSR_prev:.6f}, SEE={SEE_prev:.6f}")
        logger.info(f"Initial SSR={SSR_prev:.6f}, SEE={SEE_prev:.6f}")

        # Iterations
        for it in range(1, max_iters+1):
            print(f"[p_opt] Iter {it}")
            # CVX variable
            p_var = cp.Variable(K, nonneg=True)
            # Concave surrogate for SSR
            SSR_surr = self.power_utils_cls.SSR_active_concave_p(gamma, p_var, p_prev, cvx_bool=0)
            # Objective
            if not opt_bool:
                objective = SSR_surr
            else:
                objective = SSR_surr - ln_prev * cp.sum(cp.multiply(mu_eq , p_var))
            # Constraints
            constraints = [cp.sum(p_var) <= self.Ptmax]
            # Solve
            t0 = time.perf_counter()
            prob = cp.Problem(cp.Maximize(objective), constraints)
            try:
                prob.solve(solver='MOSEK', warm_start=True)
            except cp.error.SolverError as e:
                print(f"MOSEK failed: {e}. Falling back to SCS.")
                logger.warning("MOSEK solver error, using SCS.")
                prob.solve(solver='SCS', warm_start=True)
            t1 = time.perf_counter()
            iter_time = t1 - t0
            logger.info(f"Power iterate {it} solved in {iter_time:.4f}s, status={prob.status}")
            print(f"[p_opt] Solver status={prob.status}, time={iter_time:.4f}s")

            # Retrieve and evaluate
            p_curr = p_var.value
            SSR_expr = self.power_utils_cls.SSR_active_concave_p(gamma, p_curr, p_prev, cvx_bool=1)
            SSR_curr = max(SSR_expr.value, 0)
            denom_curr = mu_eq @ p_curr + Pc_eq
            SEE_curr = SSR_curr / denom_curr
            ln_curr = SSR_curr if not opt_bool else SEE_curr
            print(f"[p_opt] SSR={SSR_curr:.6f}, SEE={SEE_curr:.6f}")

            # Convergence
            if abs(ln_curr - ln_prev) < tol and prob.status.startswith('optimal'):
                print(f"[p_opt] Converged: Δ={ln_curr-ln_prev:.2e} < tol={tol}")
                logger.info("Convergence reached in power optimization.")
                # Safeguard: objective decrease
                if ln_curr < ln_prev:
                    print("[p_opt] Objective decreased; reverting to previous p.")
                    logger.info("Objective decreased; using previous iterate.")
                    p_curr = p_prev
                    SSR_curr, SEE_curr = SSR_prev, SEE_prev
                    break
              
                p_prev = p_curr
                SSR_prev, SEE_prev = SSR_curr, SEE_curr
                break

            # Update for next
            p_prev = p_curr.copy()
            SSR_prev, SEE_prev, ln_prev = SSR_curr, SEE_curr, ln_curr

        total_time = time.perf_counter() - start_all
        print(f"[p_opt] Completed in {it} iters, total_time={total_time:.3f}s, SSR_opt={SSR_prev:.6f}, SEE_opt={SEE_prev:.6f}")
        logger.info(f"[Power_opt_algo1] Done. iters={it} -- SSR={SSR_prev:.6f}, SEE={SEE_prev:.6f}, time={total_time:.3f}s")

        return p_prev, SSR_prev, SEE_prev, it, total_time

    # def p_cvxopt_algo1_mod(self, gamma, p, opt_bool, ris_state):
    #     """
    #     Optimize power allocation using the CVX optimization algorithm.

    #     Parameters:
    #     - gamma: Current reflection coefficients.
    #     - p: Initial power allocation vector for UEs.
    #     - opt_bool: Boolean indicating whether to optimize energy efficiency.
    #     - ris_state: State of RIS ('active' or 'passive').

    #     Returns:
    #     - p_sol: Optimized power allocation.
    #     - SSR_approx_sol: Secrecy sum rate approximation solution.
    #     - SEE_approx_sol: Secrecy energy efficiency approximation solution.
    #     - iter: Number of iterations.
    #     - tcpx_p: Time complexity for power optimization.
    #     """
    #     iter = 0
    #     p_sol = p.copy()
    #     p0 = p.copy()
    #     K = p.shape[0]
    #     SSR_approx_prev = 0
    #     SEE_approx_prev = 0
    #     ln_prev = 0

    #     SSR_approx_nxt = self.power_utils_cls.SSR_active_concave_p(gamma, p_sol, p0, cvx_bool=1)
    #     SSR_approx_nxt = max(SSR_approx_nxt.value, 0)
    #     Pc_eq = self.power_utils_cls.compute_Pc_eq_p(gamma)

    #     if ris_state == 'active':
    #         mu_eq = self.power_utils_cls.compute_mu_eq_p(gamma)
    #     else:
    #         mu_eq = np.ones_like(p) * self.mu

    #     SEE_approx_nxt = SSR_approx_nxt / (mu_eq @ p.T + Pc_eq)

    #     if opt_bool == 0:
    #         ln_nxt = SSR_approx_nxt
    #     else:
    #         ln_nxt = SEE_approx_nxt

    #     status = "unknown"
    #     tol = 1e-3
    #     flag = False

    #     if ln_nxt < tol:
    #         temp = ln_nxt
    #         ln_nxt = 10 * tol
    #         flag = True

    #     start_time = time.time()

    #     while ln_nxt - ln_prev > tol or (status not in ["optimal", "optimal_inaccurate"]):
    #         if flag:
    #             ln_nxt = temp
    #             flag = False

    #         iter += 1
    #         SSR_approx_prev = SSR_approx_nxt
    #         SEE_approx_prev = SEE_approx_nxt
    #         ln_prev = ln_nxt
    #         p0 = p_sol.copy()

    #         p_opt = cp.Variable(K, nonneg=True)
    #         objective = self.power_utils_cls.SSR_active_concave_p(gamma, p_opt, p0, cvx_bool=0) - opt_bool * ln_prev * cp.sum((cp.matmul(mu_eq, p_opt.T)))

    #         constraints = [
    #             cp.sum(p_opt) <= self.Ptmax,
    #             p_opt >= 0
    #         ]

    #         problem = cp.Problem(cp.Maximize(objective), constraints)
    #         # problem.solve(solver='MOSEK', warm_start=True)
    #         try:
    #             problem.solve(solver='MOSEK', warm_start=True)
    #         except cp.error.SolverError as e:
    #             print("MOSEK failed, trying another solver. Error:", e)
    #             problem.solve(solver='SCS', warm_start=True)

    #         p_sol = p_opt.value

    #         SSR_approx_nxt = self.power_utils_cls.SSR_active_concave_p(gamma, p_sol, p0, cvx_bool=1)
    #         SSR_approx_nxt = max(SSR_approx_nxt.value, 0)
    #         SEE_approx_nxt = SSR_approx_nxt / (mu_eq @ p_sol.T + Pc_eq)

    #         status = problem.status
    #         solver_time = problem.solver_stats.solve_time

    #         print(f'popt step: {iter}, popt_tot: {np.sum(p_sol)}, SSR_approx_nxt: {SSR_approx_nxt}, SEE_approx_nxt: {self.BW * SEE_approx_nxt}, Solver time: {solver_time}, Solver status: {status}')

    #         # if opt_bool == 0:
    #         #     ln_nxt = SSR_approx_nxt
    #         # else:
    #         #     ln_nxt = SEE_approx_nxt
                
    #         ln_nxt = SSR_approx_nxt if opt_bool == 0 else SEE_approx_nxt

    #     elapsed_time = time.time() - start_time
    #     tcpx_p = elapsed_time

    #     SSR_approx_sol = SSR_approx_nxt
    #     SEE_approx_sol = self.BW * SEE_approx_nxt

    #     if ln_nxt < ln_prev:
    #         print('\nStopping Optimization of p... optimal point reached!')
    #         p_sol = p0
    #         SSR_approx_sol = SSR_approx_prev
    #         SEE_approx_sol = self.BW * SEE_approx_prev

    #     print(f'\npopt final -> Number of Steps: {iter}, popt_tot: {np.sum(p_sol)}, SSR_opt: {SSR_approx_sol}, SEE_opt: {SEE_approx_sol}, time_complexity_p: {tcpx_p}, Solver status: {status}\n')
    #     return p_sol, SSR_approx_sol, SEE_approx_sol, iter, tcpx_p

    
    def altopt_algo1(
        self,
        gamma_init: np.ndarray,
        p_init: np.ndarray,
        opt_bool: bool,
        rf_state: str,
        ris_state: str,
        cons_state: str,
        gamma_method: str,
        bits_range: Tuple[Tuple[int,int], ...],
        quantization: bool = False,
        max_alt_iters: int = 20,
        tol: float = 1e-3
        ) -> Tuple[
            np.ndarray,                    # p_sol
            np.ndarray,                    # gamma_sol
            Dict[str, np.ndarray],         # gamma_quantized
            float, float,                  # sr, sr_q
            float, float,                  # ssr, ssr_q
            float, float,                  # gee, gee_q
            float, float,                  # see, see_q
            int,                           # alt_iters
            float, float, float, float, float  # avg_times and its
        ]:
        """
        Cleaned alternating optimization (Algorithm 1 Modified) with detailed logging.
        """
       
        logger = logging.getLogger(__name__)
        print("[altopt] Starting alternating optimization...")
        logger.info("Alternating optimization initiated.")

        # Initialize variables
        K = p_init.shape[0]
        N = gamma_init.shape[0]
        p_sol = p_init.copy()
        gamma_sol = gamma_init.copy()
        GE_true = self.G_E + (self.GE_error if self.scsi_bool == 1 else 0)
        ln_prev = -np.inf
        total_p_time = 0.0
        total_g_time = 0.0
        total_iters_p = 0
        total_iters_g = 0
        start_alt = time.time()

        Pc_eq = self.gamma_utils_cls.compute_Pc_eq_gamma(p_sol)
        R = self.utils_cls.compute_R(self.H, p_sol, self.sigma_RIS_sq)

        for alt_it in range(1, max_alt_iters + 1):
            # compute current metrics
            ssr = max(
                self.utils_cls.SR_active_algo1(self.G_B, self.H, gamma_sol, p_sol,
                                            self.sigma_sq, self.sigma_RIS_sq,
                                            self.sigma_e_sq, self.scsi_bool,
                                            True, "Bob") -
                self.utils_cls.SR_active_algo1(GE_true, self.H, gamma_sol, p_sol,
                                            self.sigma_sq, self.sigma_RIS_sq,
                                            self.sigma_e_sq, self.scsi_bool,
                                            True, "Eve"),
                0
            )
            see = self.BW * max(
                self.utils_cls.GEE_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.scsi_bool, True, "Bob") -
                self.utils_cls.GEE_active_algo1(GE_true, self.H, gamma_sol, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.scsi_bool, True, "Eve"), 0
            )
            
            # ssr / (ris_state == 'active' and (self.a - 1 if rf_state == 'RF-Gain' else self.PRmax) + Pc_eq)
            print(f"[altopt] Iter {alt_it}: SSR={ssr:.3e}, SEE={see:.6f}")
            logger.info(f"Iteration {alt_it} start: SSR={ssr:.3e}, SEE={see:.6f}")

            # convergence check
            ln_curr = ssr if not opt_bool else see / self.BW
            if alt_it > 1 and abs(ln_curr - ln_prev) / max(ln_prev,  1e-12) < tol:
                print(f"[altopt] Converged: Δ={ln_curr-ln_prev:.3e} < tol={tol}")
                logger.info("Convergence reached.")
                break
            ln_prev = ln_curr

            # gamma optimization
            print(f"[altopt] Gamma optimization ({gamma_method})...")
            logger.info(f"Gamma optimization method: {gamma_method}")
            t0 = time.time()
            if gamma_method == 'cvx1':
                gamma_sol, ssr_g, see_g, it_g, t_g = self.gamma_cvxopt_algo1(
                    gamma_sol, p_sol, opt_bool, rf_state, ris_state, cons_state, max_iters=50, tol=tol)
            elif gamma_method == 'cvx2':
                gamma_sol, ssr_g, see_g, it_g, t_g = self.gamma_cvxopt_algo2(
                    gamma_init= gamma_sol, p=p_sol, opt_bool=opt_bool, rf_state=rf_state,
                    ris_state=ris_state, cons_state=cons_state,
                    extraction='pca', num_randomizations=50,
                    max_sca_iters=30, tol=tol)
            elif gamma_method == 'ls':
                gamma_sol, ssr_g, see_g, it_g, t_g = self.optimize_gamma_ls(
                    gamma_sol, p_sol, opt_bool, rf_state,
                    ris_state, cons_state, tol=tol, max_cycles=30)
            else:
                raise ValueError(f"Unknown gamma_method '{gamma_method}'")
            total_iters_g += it_g
            total_g_time += t_g
            print(f"[altopt] Gamma done: its={it_g}, time={t_g:.3f}s, SSR_g={ssr_g:.3e}, SEE_g={self.BW * see_g:.6f}")
            logger.info(f"Gamma opt iters={it_g}, time={t_g:.4f}s")

            # power optimization
            print("[altopt] Power optimization...")
            logger.info("Power optimization start.")
            t1 = time.time()
            p_sol, ssr_p, see_p, it_p, t_p = self.p_cvxopt_algo1(
                gamma_sol, p_sol, opt_bool, ris_state, max_iters=50, tol=tol)
            total_iters_p += it_p
            total_p_time += t_p
            print(f"[altopt] Power done: its={it_p}, time={t_p:.3f}s, SSR_p={ssr_p:.3e}, SEE_p={self.BW * see_p:.6f}")
            logger.info(f"Power opt iters={it_p}, time={t_p:.4f}s")

        total_alt = time.time() - start_alt
        print(f"[altopt] Alt optimization complete in {total_alt:.3f}s")
        logger.info(f"Alt optimization finished in {total_alt:.3f}s")

        # final metrics
        sr = self.utils_cls.SR_active_algo1(self.G_B, self.H,
            gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq,
            self.sigma_e_sq, self.scsi_bool, True, "Bob")
        gee = self.BW * self.utils_cls.GEE_active_algo1(self.G_B, self.H,
            gamma_sol, p_sol, self.mu, self.Pc,
            self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq,
            ris_state, self.scsi_bool, True, "Bob")

        quant = {'Gamma': gamma_sol, 'SR': sr, 'SSR': ssr, 'GEE': gee, 'SEE': see}
        if quantization:
            print("[altopt] Applying quantization...")
            logger.info("Quantization start.")
            # Optimized Input power matrix at the RIS 
            R = self.utils_cls.compute_R(self.H, p_sol, self.sigma_RIS_sq)
            
            # Ensure R is positive definite
            assert np.all(np.linalg.eigvals(R) > 0), "Matrix R is not positive definite."

            # Compute the eigenvalues of R
            eigenvalues_R = np.linalg.eigvalsh(R)

            # Find the minimum and maximum eigenvalue
            min_eigenvalue_R = np.min(eigenvalues_R)
            max_eigenvalue_R = np.max(eigenvalues_R)
            
            if ris_state == 'active':
                PRmax = (self.a - 1)* np.real(np.trace(R))  if rf_state == "RF-Gain" else self.PRmax
            else:
                PRmax = 0

            # Compute the minimum / maximum value for each element of gamma in the equal elements case      
            a_min =  np.sqrt((np.real(np.trace(R))) / (N * max_eigenvalue_R)) if ris_state == 'active' else 0
            
            a_max =  np.sqrt((PRmax + np.real(np.trace(R))) /  (N * min_eigenvalue_R)) if cons_state == 'global' else 1

            
            quant = self.optimize_with_quantization(gamma_sol, p_sol, ris_state, a_min, a_max, bits_range)
            
            # gamma_sol_Q = quant['Gamma']
            # sr_sol_Q =  quant['SR']
            # ssr_sol_Q = quant['SSR']
            # gee_sol_Q = quant['GEE']
            # see_sol_Q = quant['SEE']
            
            gamma_sol_norm_Q_str = ", ".join([f"{key}: {np.linalg.norm(val)}" for key, val in quant['Gamma'].items()])
            SR_opt_Q_str = ", ".join([f"{key}: {val:.3e}" for key, val in quant['SR'].items()])
            SSR_opt_Q_str = ", ".join([f"{key}: {val:.3e}" for key, val in quant['SSR'].items()])
            GEE_opt_Q_str = ", ".join([f"{key}: {val:.6f}" for key, val in quant['GEE'].items()])
            SEE_opt_Q_str = ", ".join([f"{key}: {val:.6f}" for key, val in quant['SEE'].items()])

        print("[altopt] Final results:")
        print(f"  p: sum={np.sum(p_sol):.3e}, gamma_norm={np.linalg.norm(gamma_sol):.3e}, gamma_norm_Q_str={gamma_sol_norm_Q_str}")
        print(f" SR={SR_opt_Q_str}, SSR={SSR_opt_Q_str}, GEE={GEE_opt_Q_str}, SEE={SEE_opt_Q_str}")
        logger.info(f"Final SR={SR_opt_Q_str}, SSR={SSR_opt_Q_str}, GEE={GEE_opt_Q_str}, SEE={SEE_opt_Q_str}")

        return p_sol, gamma_sol, quant['Gamma'], sr, quant['SR'], ssr, quant['SSR'], gee, quant['GEE'], see, quant['SEE'], alt_it, total_iters_p / alt_it, total_iters_g / alt_it, total_alt, total_p_time / alt_it, total_g_time / alt_it
               


    # def altopt_algo1_mod(self, gamma, p, opt_bool, rf_state, ris_state, cons_state, gamma_method, bits_range, quantization):
    #     """
    #     Perform alternating optimization for gamma and power allocation.

    #     Parameters:
    #     - gamma: Initial reflection coefficients.
    #     - p: Initial power allocation vector for UEs.
    #     - bits_phase: Number of bits for phase quantization.
    #     - bits_amplitude: Number of bits for amplitude quantization.
    #     - quantization: Boolean indicating whether to perform quantization.
    #     - ris_state: State of RIS ('active' or 'passive').
    #     - cons_state: Constraint state ('global' or 'local').
    #     - opt_bool: Boolean indicating whether to optimize energy efficiency.

    #     Returns:
    #     - Optimized values for power, gamma, SSR, SEE, and related metrics.
    #     """
    #     K = p.shape[0]
    #     N = gamma.shape[0]
    #     # ssr_prev = 0
    #     # see_prev = 0
    #     ln_prev = 0
    #     p_sol = p.copy()
    #     gamma_sol = gamma.copy()
    #     iteration_altopt = 0
    #     iteration_gamma = 0
    #     iteration_p = 0
    #     time_complexity_gamma = 0
    #     time_complexity_p = 0

    #     # GE_true = self.G_E + np.zeros_like(self.G_E)  # Assuming gE_error is zero
    #     # GE_true = self.G_E + self.GE_error if self.scsi_bool == 0 else self.G_E
    #     csi_bool = True if self.scsi_bool == 0 else False
    #     # G_E = self.G_E + self.GE_error if self.scsi_bool == 0 else self.G_E 
    #     GE_true = self.G_E + self.GE_error if self.scsi_bool == 1 else self.G_E
        
    #     ssr_nxt = max(
    #         self.utils_cls.SR_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.scsi_bool, True, "Bob") -
    #         self.utils_cls.SR_active_algo1(GE_true, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.scsi_bool, True, "Eve"), 0
    #     ) # csi_bool
    #     see_nxt = max(
    #         self.utils_cls.GEE_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.scsi_bool, True, "Bob") -
    #         self.utils_cls.GEE_active_algo1(GE_true, self.H, gamma_sol, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.scsi_bool, True, "Eve"), 0
    #     ) # csi_bool

    #     ln_nxt = ssr_nxt if opt_bool == 0 else see_nxt
        
    #     # if opt_bool == 0:
    #     #     ln_nxt = ssr_nxt
    #     # else:
    #     #     ln_nxt = see_nxt

    #     tol = 1e-3
    #     flag = False

    #     if ln_nxt < tol:
    #         temp = ln_nxt
    #         ln_nxt = 10 * tol
    #         flag = True

    #     start_time = time.time()

    #     while ln_nxt - ln_prev > tol:
    #         if flag:
    #             ln_nxt = temp
    #             flag = False

    #         iteration_altopt += 1

    #         # ssr_prev = ssr_nxt
    #         # see_prev = see_nxt
    #         ln_prev = ln_nxt
    #         p_prev = p_sol.copy()
    #         gamma_prev = gamma_sol.copy()
            
    #         # ───── gamma optimization ────────────────────────────────────────────────────
    #         if gamma_method in ('cvx1', 'cvx2'):
    #             if gamma_method == 'cvx1':
    #                 # CVX-based Algorithm 1
    #                 gamma_sol, SSR_opt, SEE_opt, iter_gamma, tcpx_gamma = \
    #                     self.gamma_cvxopt_algo1_mod(
    #                         gamma_prev,    # initial gamma from last outer step
    #                         p_sol,         # power allocation
    #                         opt_bool,      # SSR vs SEE
    #                         rf_state,
    #                         ris_state,
    #                         cons_state,
    #                         max_iters=20,
    #                         tol=1e-3
    #                     )
    #             else:  # 'cvx2'
    #                 # SDR + SCA (Algorithm 2)
    #                 gamma_sol, SSR_opt, SEE_opt, iter_gamma, tcpx_gamma = \
    #                     self.gamma_opt_algo2(
    #                         p=p_sol,
    #                         opt_bool=opt_bool,
    #                         rf_state=rf_state,
    #                         ris_state=ris_state,
    #                         cons_state=cons_state,
    #                         extraction='pca',            # or 'randomization'
    #                         num_randomizations=20,
    #                         max_sca_iters=10,
    #                         tol=1e-3
    #                     )
    #         elif gamma_method == 'ls':
    #             # Low-complexity line-search
    #             gamma_sol, SSR_opt, SEE_opt, iter_gamma, tcpx_gamma = \
    #                 self.optimize_gamma_ls(
    #                     gamma_prev,
    #                     p_sol,
    #                     opt_bool,
    #                     rf_state,
    #                     ris_state,
    #                     cons_state,
    #                     tol=1e-3,
    #                     max_cycles=50
    #                 )
    #         else:
    #             raise ValueError(f"Unknown gamma_method '{gamma_method}'. "
    #                             "Use 'cvx1','cvx2' or 'ls'.")
    #         # ───────────────────────────────────────────────────────────────────────────────

    #         # gamma optimization
    #         # if gamma_method == 'cvx':
    #         #     gamma_sol, _, _, iter_gamma, tcpx_gamma = self.gamma_cvxopt_algo1_mod(
    #         #     gamma_prev, p_sol, opt_bool, rf_state, ris_state, cons_state)
    #         # else:
    #         #     gamma_sol, _, _, iter_gamma, tcpx_gamma= self.optimize_gamma_ls(
    #         #         gamma_prev, p_sol, opt_bool, rf_state, ris_state, cons_state,
    #         #         tol=1e-3, max_cycles=50)

    #         # gamma_sol, _, _, iter_gamma, tcpx_gamma = self.gamma_cvxopt_algo1_mod(
    #         #     gamma_prev, p_sol, opt_bool, rf_state, ris_state, cons_state)

    #         # power optimization
    #         p_sol, _, _, iter_p, tcpx_p = self.p_cvxopt_algo1_mod(
    #             gamma_sol, p_prev, opt_bool, ris_state)

    #         ssr_nxt = max(
    #             self.utils_cls.SR_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.scsi_bool, True, "Bob") -
    #             self.utils_cls.SR_active_algo1(self.G_E, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.scsi_bool, csi_bool, "Eve"), 0
    #         )
    #         see_nxt = max(
    #             self.utils_cls.GEE_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.scsi_bool, True, "Bob") -
    #             self.utils_cls.GEE_active_algo1(self.G_E, self.H, gamma_sol, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.scsi_bool, csi_bool, "Eve"), 0
    #         )

    #         print(f"Alternating Optimization step: {iteration_altopt}, p_sol_tot: {np.sum(p_sol)}, gamma_sol_norm: {np.linalg.norm(gamma_sol)}, SSR_nxt: {ssr_nxt}, SEE_nxt: {self.BW * see_nxt}\n")

    #         ln_nxt  = ssr_nxt if opt_bool == 0 else see_nxt
    #         # if opt_bool == 0:
    #         #     ln_nxt = ssr_nxt
    #         # else:
    #         #     ln_nxt = see_nxt

    #         iteration_gamma += iter_gamma
    #         iteration_p += iter_p
    #         time_complexity_gamma += tcpx_gamma
    #         time_complexity_p += tcpx_p

    #     elapsed_time = time.time() - start_time
    #     time_complexity_altopt = elapsed_time

    #     if ln_nxt < ln_prev:
    #         print('\nStopping Alternating Optimization... optimal point reached!')
    #         p_sol = p_prev
    #         gamma_sol = gamma_prev
        
    #     sr_sol =  self.BW * self.utils_cls.SR_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.scsi_bool, True, "Bob")
    #     ssr_sol = max(
    #         self.utils_cls.SR_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.scsi_bool, True, "Bob") -
    #         self.utils_cls.SR_active_algo1(GE_true, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.scsi_bool, True, "Eve"), 0
    #     )
    #     gee_sol =  self.BW * self.utils_cls.GEE_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.scsi_bool, True, "Bob")
    #     see_sol = self.BW * max(
    #         self.utils_cls.GEE_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.scsi_bool, True, "Bob") -
    #         self.utils_cls.GEE_active_algo1(GE_true, self.H, gamma_sol, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.scsi_bool, True, "Eve"), 0
    #     )
                        
    #     if quantization:
    #         # Optimized Input power matrix at the RIS 
    #         R = self.utils_cls.compute_R(self.H, p_sol, self.sigma_RIS_sq)
            
    #         # Ensure R is positive definite
    #         assert np.all(np.linalg.eigvals(R) > 0), "Matrix R is not positive definite."

    #         # Compute the eigenvalues of R
    #         eigenvalues_R = np.linalg.eigvalsh(R)

    #         # Find the minimum and maximum eigenvalue
    #         min_eigenvalue_R = np.min(eigenvalues_R)
    #         max_eigenvalue_R = np.max(eigenvalues_R)
            
    #         if ris_state == 'active':
    #             PRmax = (self.a - 1)* np.real(np.trace(R))  if rf_state == "RF-Gain" else self.PRmax
    #         else:
    #             PRmax = 0
            
    #         # Compute the maximum value for the single non-zero element of gamma
    #         # max_gamma_value_single = np.sqrt((P_Rmax + np.trace(R)) / min_eigenvalue)
            
    #         # Compute the minimum / maximum value for each element of gamma in the equal elements case
            
    #         a_min =  np.sqrt((np.real(np.trace(R))) / (N * max_eigenvalue_R)) if ris_state == 'active' else 0
            
    #         # a_min =  max(np.sqrt((np.real(np.trace(R))) / (N * max_eigenvalue_R)), 1) if ris_state == 'active' else 0
            
    #         a_max =  np.sqrt((PRmax + np.real(np.trace(R))) /  (N * min_eigenvalue_R)) if cons_state == 'global' else 1

    #         quantization_results = self.optimize_with_quantization(gamma_sol, p_sol, ris_state, a_min, a_max, bits_range)
        
    #         gamma_sol_Q = quantization_results['Gamma']
    #         sr_sol_Q = quantization_results['SR']
    #         ssr_sol_Q = quantization_results['SSR']
    #         gee_sol_Q = quantization_results['GEE']
    #         see_sol_Q = quantization_results['SEE']
            
    #         gamma_sol_norm_Q_str = ", ".join([f"{key}: {np.linalg.norm(val)}" for key, val in quantization_results['Gamma'].items()])
    #         SR_opt_Q_str = ", ".join([f"{key}: {val}" for key, val in quantization_results['SR'].items()])
    #         SSR_opt_Q_str = ", ".join([f"{key}: {val}" for key, val in quantization_results['SSR'].items()])
    #         GEE_opt_Q_str = ", ".join([f"{key}: {val}" for key, val in quantization_results['GEE'].items()])
    #         SEE_opt_Q_str = ", ".join([f"{key}: {val}" for key, val in quantization_results['SEE'].items()])
    #     else:
            
    #         gamma_sol_Q = gamma_sol
    #         sr_sol_Q = sr_sol
    #         ssr_sol_Q = ssr_sol
    #         gee_sol_Q = gee_sol
    #         see_sol_Q = see_sol
            
    #         gamma_sol_norm_Q_str = np.linalg.norm(gamma_sol)
    #         SR_opt_Q_str = sr_sol
    #         SSR_opt_Q_str = ssr_sol
    #         GEE_opt_Q_str = gee_sol
    #         SEE_opt_Q_str = see_sol
        
            
        
    #     # if quantization:
    #     #     gamma_sol_Q = self.utils_cls.project_to_quantized_levels(gamma_sol, a_max, bits_phase, bits_amplitude)
    #     #     ssr_sol_Q = max(
    #     #         self.gamma_utils_cls.SR_active_algo1(self.G_B, self.H, gamma_sol_Q, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.scsi_bool, True, "Bob") -
    #     #         self.gamma_utils_cls.SR_active_algo1(GE_true, self.H, gamma_sol_Q, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, self.scsi_bool, True, "Eve"), 0
    #     #     )
    #     #     see_sol_Q = self.BW * max(
    #     #         self.utils_cls.GEE_active_algo1(self.G_B, self.H, gamma_sol_Q, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.scsi_bool, True, "Bob") -
    #     #         self.utils_cls.GEE_active_algo1(GE_true, self.H, gamma_sol_Q, p_sol, self.mu, self.Pc, self.sigma_sq, self.sigma_RIS_sq, self.sigma_e_sq, ris_state, self.csi_bool, True, "Eve"), 0
    #     #     )

    #     iteration_gamma /= iteration_altopt
    #     iteration_p /= iteration_altopt
    #     time_complexity_gamma /= iteration_altopt
    #     time_complexity_p /= iteration_altopt

    #     print("\n" + "="*40 + " QUANTIZATION ON " + "="*40) if quantization else print("\n" + "="*40 + " QUANTIZATION OFF " + "="*40)
    #     # print("\n****************************************************************************************************************************")
    #     print(f"Alternating Optimization -> Number of Steps: {iteration_altopt}, p_sol_tot: {np.sum(p_sol)}, gamma_sol_norm: {np.linalg.norm(gamma_sol)}, gamma_sol_norm_Q: {gamma_sol_norm_Q_str}, SR_opt: {sr_sol}, SR_opt_Q: {SR_opt_Q_str}, SSR_opt: {ssr_sol}, SSR_opt_Q: {SSR_opt_Q_str}, GEE_opt: {gee_sol}, GEE_opt_Q: {GEE_opt_Q_str}, SEE_opt: {see_sol}, SEE_opt_Q: {SEE_opt_Q_str}, iteration_altopt: {iteration_altopt}, iteration_p: {iteration_p}, iteration_gamma: {iteration_gamma}, time_complexity_altopt: {time_complexity_altopt}, time_complexity_p: {time_complexity_p}, time_complexity_gamma: {time_complexity_gamma}")
    #     print("="*100 + "\n")
    #     # print("****************************************************************************************************************************\n")

    #     return p_sol, gamma_sol, gamma_sol_Q, sr_sol, sr_sol_Q, ssr_sol, ssr_sol_Q, gee_sol, gee_sol_Q, see_sol, see_sol_Q, iteration_altopt, iteration_p, iteration_gamma, time_complexity_altopt, time_complexity_p, time_complexity_gamma
