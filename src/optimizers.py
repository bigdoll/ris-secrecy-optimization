import numpy as np
import cvxpy as cp
import time
from utils import Utils
from gamma_utils import GammaUtils
from power_utils import PowerUtils

class Optimizer:
    def __init__(self, H, G_B, G_E, GE_error, sigma_sq, sigma_RIS_sq, sigma_g_sq, mu, Pc, a, Pmax, BW, scsi_bool=1, utils_cls=Utils, gamma_utils_cls=GammaUtils, power_utils_cls=PowerUtils):
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
        self.G_E = G_E
        self.GE_error = GE_error
        self.sigma_sq = sigma_sq
        self.sigma_RIS_sq = sigma_RIS_sq
        self.sigma_g_sq = sigma_g_sq
        self.mu = mu
        self.Pc = Pc
        self.a = a
        self.Pmax = Pmax
        self.BW = BW
        self.scsi_bool = scsi_bool
        self.utils_cls = utils_cls
        self.gamma_utils_cls = gamma_utils_cls(H, G_B, G_E, sigma_sq, sigma_RIS_sq, sigma_g_sq, mu, Pc, scsi_bool)
        self.power_utils_cls = power_utils_cls(H, G_B, G_E, sigma_sq, sigma_RIS_sq, sigma_g_sq, mu, Pc, scsi_bool)

    def gamma_cvxopt_algo1_mod(self, gamma, p, opt_bool, ris_state, cons_state):
        """
        Optimize gamma using the CVX optimization algorithm.

        Parameters:
        - gamma: Initial reflection coefficients.
        - p: Power allocation vector for UEs.
        - opt_bool: Boolean indicating whether to optimize energy efficiency.
        - ris_state: State of RIS ('active' or 'passive').
        - cons_state: Constraint state ('global' or 'local').

        Returns:
        - gamma_sol: Optimized reflection coefficients.
        - SSR_approx_sol: Secrecy sum rate approximation solution.
        - SEE_approx_sol: Secrecy energy efficiency approximation solution.
        - iter: Number of iterations.
        - tcpx_gamma: Time complexity for gamma optimization.
        """
        ris_bool = 1 if ris_state == 'active' else 0
        iter = 0
        gamma_sol = gamma.copy()
        gamma_0 = gamma.copy()
        N = gamma.shape[0]
        SSR_approx_prev = 0
        SEE_approx_prev = 0
        ln_prev = 0

        R = self.utils_cls.compute_R(self.H, p, self.sigma_RIS_sq)
        Rn = np.linalg.norm(R, 'fro')
        Rnorm = R / Rn

        if ris_state == 'active':
            PRmax = self.a * np.real(np.trace(Rnorm))
        else:
            PRmax = 0

        SR_approx_nxt_Bob = self.gamma_utils_cls.SR_active_concave_gamma_Bob(gamma_sol, gamma_0, p, True)
        SR_approx_nxt_Eve = self.gamma_utils_cls.SR_active_concave_gamma_Eve(gamma_sol, gamma_0, p, True)
        SSR_approx_nxt = max(SR_approx_nxt_Bob + SR_approx_nxt_Eve, 0)
        Pc_eq = self.gamma_utils_cls.compute_Pc_eq_gamma(p)
        SEE_approx_nxt = SSR_approx_nxt / (ris_bool * np.real(np.trace(R @ (gamma_sol @ gamma_sol.conj().T))) + Pc_eq)

        if opt_bool == 0:
            ln_nxt = SSR_approx_nxt
        else:
            ln_nxt = SEE_approx_nxt

        status = "unknown"
        tol = 1e-3
        flag = False

        if ln_nxt < tol:
            temp = ln_nxt
            ln_nxt = 10 * tol
            flag = True

        start_time = time.time()

        while ln_nxt - ln_prev > tol or (status not in ["optimal", "optimal_inaccurate"]):
            if flag:
                ln_nxt = temp
                flag = False

            iter += 1
            SSR_approx_prev = SSR_approx_nxt
            SEE_approx_prev = SEE_approx_nxt
            ln_prev = ln_nxt
            gamma_0 = gamma_sol.copy()

            gamma_opt = cp.Variable((N, 1), complex=True)
            objective = self.gamma_utils_cls.SR_active_concave_gamma_Bob(gamma_opt, gamma_0, p, False) + \
                        self.gamma_utils_cls.SR_active_concave_gamma_Eve(gamma_opt, gamma_0, p, False) - \
                        ris_bool * opt_bool * ln_prev * cp.real(cp.sum(cp.quad_form(gamma_opt, R)))

            if ris_state == 'active':
                constraints = [
                    cp.real(gamma_0.conj().T @ Rnorm @ gamma_0) + 2 * cp.real(gamma_0.conj().T @ Rnorm @ (gamma_opt - gamma_0)) >= cp.real(cp.trace(Rnorm)),
                    cp.real(cp.quad_form(gamma_opt, Rnorm)) <= PRmax + cp.real(cp.trace(Rnorm))
                ]
            else:
                if cons_state == "global":
                    constraints = [cp.real(cp.quad_form(gamma_opt, Rnorm)) <= cp.real(cp.trace(Rnorm))]
                else:
                    constraints = [cp.square(cp.abs(gamma_opt)) <= self.a]

            problem = cp.Problem(cp.Maximize(objective), constraints)
            problem.solve(solver='MOSEK', warm_start=True)

            gamma_sol = gamma_opt.value

            SR_approx_nxt_Bob = self.gamma_utils_cls.SR_active_concave_gamma_Bob(gamma_sol, gamma_0, p, True)
            SR_approx_nxt_Eve = self.gamma_utils_cls.SR_active_concave_gamma_Eve(gamma_sol, gamma_0, p, True)
            SSR_approx_nxt = max(SR_approx_nxt_Bob + SR_approx_nxt_Eve, 0)
            SEE_approx_nxt = SSR_approx_nxt / (ris_bool * np.real(np.trace(R @ (gamma_sol @ gamma_sol.conj().T))) + Pc_eq)

            status = problem.status
            solver_time = problem.solver_stats.solve_time

            print(f'gamma_opt step: {iter}, gamma_nxt_norm: {np.linalg.norm(gamma_sol)}, SSR_approx_nxt: {SSR_approx_nxt}, SEE_approx_nxt: {self.BW * SEE_approx_nxt}, Solver time: {solver_time}, Solver status: {status}')

            if opt_bool == 0:
                ln_nxt = SSR_approx_nxt
            else:
                ln_nxt = SEE_approx_nxt

        elapsed_time = time.time() - start_time
        tcpx_gamma = elapsed_time

        SSR_approx_sol = SSR_approx_nxt
        SEE_approx_sol = self.BW * SEE_approx_nxt

        if ln_nxt < ln_prev:
            print('\nStopping Optimization of gamma... optimal point reached!')
            gamma_sol = gamma_0
            SSR_approx_sol = SSR_approx_prev
            SEE_approx_sol = self.BW * SEE_approx_prev

        print(f'\ngamma_opt final -> Number of Steps: {iter}, SSR_opt: {SSR_approx_sol}, SEE_opt: {SEE_approx_sol}, time_complexity_gamma: {tcpx_gamma}, Solver status: {status}\n')
        return gamma_sol, SSR_approx_sol, SEE_approx_sol, iter, tcpx_gamma

    def p_cvxopt_algo1_mod(self, gamma, p, opt_bool, ris_state):
        """
        Optimize power allocation using the CVX optimization algorithm.

        Parameters:
        - gamma: Current reflection coefficients.
        - p: Initial power allocation vector for UEs.
        - opt_bool: Boolean indicating whether to optimize energy efficiency.
        - ris_state: State of RIS ('active' or 'passive').

        Returns:
        - p_sol: Optimized power allocation.
        - SSR_approx_sol: Secrecy sum rate approximation solution.
        - SEE_approx_sol: Secrecy energy efficiency approximation solution.
        - iter: Number of iterations.
        - tcpx_p: Time complexity for power optimization.
        """
        iter = 0
        p_sol = p.copy()
        p0 = p.copy()
        K = p.shape[0]
        SSR_approx_prev = 0
        SEE_approx_prev = 0
        ln_prev = 0

        SSR_approx_nxt = self.power_utils_cls.SSR_active_concave_p(gamma, p_sol, p0, True)
        SSR_approx_nxt = max(SSR_approx_nxt, 0)
        Pc_eq = self.power_utils_cls.compute_Pc_eq_p(gamma)

        if ris_state == 'active':
            mu_eq = self.power_utils_cls.compute_mu_eq_p(gamma)
        else:
            mu_eq = np.ones_like(p) * self.mu

        SEE_approx_nxt = SSR_approx_nxt / (mu_eq @ p.T + Pc_eq)

        if opt_bool == 0:
            ln_nxt = SSR_approx_nxt
        else:
            ln_nxt = SEE_approx_nxt

        status = "unknown"
        tol = 1e-3
        flag = False

        if ln_nxt < tol:
            temp = ln_nxt
            ln_nxt = 10 * tol
            flag = True

        start_time = time.time()

        while ln_nxt - ln_prev > tol or (status not in ["optimal", "optimal_inaccurate"]):
            if flag:
                ln_nxt = temp
                flag = False

            iter += 1
            SSR_approx_prev = SSR_approx_nxt
            SEE_approx_prev = SEE_approx_nxt
            ln_prev = ln_nxt
            p0 = p_sol.copy()

            p_opt = cp.Variable(K, nonneg=True)
            objective = self.power_utils_cls.SSR_active_concave_p(gamma, p_opt, p0, False) - opt_bool * ln_prev * cp.sum((cp.matmul(mu_eq, p_opt.T)))

            constraints = [
                cp.sum(p_opt) <= self.Pmax,
                p_opt >= 0
            ]

            problem = cp.Problem(cp.Maximize(objective), constraints)
            problem.solve(solver='MOSEK', warm_start=True)

            p_sol = p_opt.value

            SSR_approx_nxt = self.power_utils_cls.SSR_active_concave_p(gamma, p_sol, p0, True)
            SSR_approx_nxt = max(SSR_approx_nxt, 0)
            SEE_approx_nxt = SSR_approx_nxt / (mu_eq @ p_sol.T + Pc_eq)

            status = problem.status
            solver_time = problem.solver_stats.solve_time

            print(f'popt step: {iter}, popt_tot: {np.sum(p_sol)}, SSR_approx_nxt: {SSR_approx_nxt}, SEE_approx_nxt: {self.BW * SEE_approx_nxt}, Solver time: {solver_time}, Solver status: {status}')

            if opt_bool == 0:
                ln_nxt = SSR_approx_nxt
            else:
                ln_nxt = SEE_approx_nxt

        elapsed_time = time.time() - start_time
        tcpx_p = elapsed_time

        SSR_approx_sol = SSR_approx_nxt
        SEE_approx_sol = self.BW * SEE_approx_nxt

        if ln_nxt < ln_prev:
            print('\nStopping Optimization of p... optimal point reached!')
            p_sol = p0
            SSR_approx_sol = SSR_approx_prev
            SEE_approx_sol = self.BW * SEE_approx_prev

        print(f'\npopt final -> Number of Steps: {iter}, popt_tot: {np.sum(p_sol)}, SSR_opt: {SSR_approx_sol}, SEE_opt: {SEE_approx_sol}, time_complexity_p: {tcpx_p}, Solver status: {status}\n')
        return p_sol, SSR_approx_sol, SEE_approx_sol, iter, tcpx_p

    def altopt_algo1_mod(self, gamma, p, bits_phase, bits_amplitude, quantization, ris_state, cons_state, opt_bool):
        """
        Perform alternating optimization for gamma and power allocation.

        Parameters:
        - gamma: Initial reflection coefficients.
        - p: Initial power allocation vector for UEs.
        - bits_phase: Number of bits for phase quantization.
        - bits_amplitude: Number of bits for amplitude quantization.
        - quantization: Boolean indicating whether to perform quantization.
        - ris_state: State of RIS ('active' or 'passive').
        - cons_state: Constraint state ('global' or 'local').
        - opt_bool: Boolean indicating whether to optimize energy efficiency.

        Returns:
        - Optimized values for power, gamma, SSR, SEE, and related metrics.
        """
        K = p.shape[0]
        N = gamma.shape[0]
        ssr_prev = 0
        see_prev = 0
        ln_prev = 0
        p_sol = p.copy()
        gamma_sol = gamma.copy()
        iteration_altopt = 0
        iteration_gamma = 0
        iteration_p = 0
        time_complexity_gamma = 0
        time_complexity_p = 0

        # GE_true = self.G_E + np.zeros_like(self.G_E)  # Assuming gE_error is zero
        GE_true = self.G_E + self.GE_error if self.scsi_bool == 0 else self.G_E

        
        ssr_nxt = max(
            self.utils_cls.SR_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, self.scsi_bool, False, "Bob") -
            self.utils_cls.SR_active_algo1(GE_true, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, self.scsi_bool, False, "Eve"), 0
        )
        see_nxt = max(
            self.utils_cls.GEE_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, ris_state, self.scsi_bool, False, "Bob") -
            self.utils_cls.GEE_active_algo1(GE_true, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, ris_state, self.scsi_bool, False, "Eve"), 0
        )

        if opt_bool == 0:
            ln_nxt = ssr_nxt
        else:
            ln_nxt = see_nxt

        tol = 1e-3
        flag = False

        if ln_nxt < tol:
            temp = ln_nxt
            ln_nxt = 10 * tol
            flag = True

        start_time = time.time()

        while ln_nxt - ln_prev > tol:
            if flag:
                ln_nxt = temp
                flag = False

            iteration_altopt += 1

            ssr_prev = ssr_nxt
            see_prev = see_nxt
            ln_prev = ln_nxt
            p_prev = p_sol.copy()
            gamma_prev = gamma_sol.copy()

            gamma_sol, _, _, iter_gamma, tcpx_gamma = self.gamma_cvxopt_algo1_mod(
                gamma_prev, p_sol, opt_bool, ris_state, cons_state)

            p_sol, _, _, iter_p, tcpx_p = self.p_cvxopt_algo1_mod(
                gamma_sol, p_prev, opt_bool, ris_state)

            ssr_nxt = max(
                self.gamma_utils_cls.SR_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, self.scsi_bool, False, "Bob") -
                self.gamma_utils_cls.SR_active_algo1(GE_true, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, self.scsi_bool, False, "Eve"), 0
            )
            see_nxt = max(
                self.gamma_utils_cls.GEE_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, ris_state, self.scsi_bool, False, "Bob") -
                self.gamma_utils_cls.GEE_active_algo1(GE_true, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, ris_state, self.scsi_bool, False, "Eve"), 0
            )

            print(f"Alternating Optimization step: {iteration_altopt}, p_sol_tot: {np.sum(p_sol)}, gamma_sol_norm: {np.linalg.norm(gamma_sol)}, SSR_nxt: {ssr_nxt}, SEE_nxt: {self.BW * see_nxt}\n")

            if opt_bool == 0:
                ln_nxt = ssr_nxt
            else:
                ln_nxt = see_nxt

            iteration_gamma += iter_gamma
            iteration_p += iter_p
            time_complexity_gamma += tcpx_gamma
            time_complexity_p += tcpx_p

        elapsed_time = time.time() - start_time
        time_complexity_altopt = elapsed_time

        if ln_nxt < ln_prev:
            print('\nStopping Alternating Optimization... optimal point reached!')
            p_sol = p_prev
            gamma_sol = gamma_prev

        ssr_sol = max(
            self.gamma_utils_cls.SR_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, self.scsi_bool, True, "Bob") -
            self.gamma_utils_cls.SR_active_algo1(GE_true, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, self.scsi_bool, True, "Eve"), 0
        )
        see_sol = self.BW * max(
            self.gamma_utils_cls.GEE_active_algo1(self.G_B, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, ris_state, self.scsi_bool, True, "Bob") -
            self.gamma_utils_cls.GEE_active_algo1(GE_true, self.H, gamma_sol, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, ris_state, self.scsi_bool, True, "Eve"), 0
        )

        gamma_sol_Q = gamma_sol
        ssr_sol_Q = ssr_sol
        see_sol_Q = see_sol
        a_max = np.sqrt(N)

        if quantization:
            gamma_sol_Q = self.utils_cls.project_to_quantized_levels(gamma_sol, a_max, bits_phase, bits_amplitude)
            ssr_sol_Q = max(
                self.gamma_utils_cls.SR_active_algo1(self.G_B, self.H, gamma_sol_Q, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, self.scsi_bool, True, "Bob") -
                self.gamma_utils_cls.SR_active_algo1(GE_true, self.H, gamma_sol_Q, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, self.scsi_bool, True, "Eve"), 0
            )
            see_sol_Q = self.BW * max(
                self.gamma_utils_cls.GEE_active_algo1(self.G_B, self.H, gamma_sol_Q, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, ris_state, self.scsi_bool, True, "Bob") -
                self.gamma_utils_cls.GEE_active_algo1(GE_true, self.H, gamma_sol_Q, p_sol, self.sigma_sq, self.sigma_RIS_sq, self.sigma_g_sq, ris_state, self.scsi_bool, True, "Eve"), 0
            )

        iteration_gamma /= iteration_altopt
        iteration_p /= iteration_altopt
        time_complexity_gamma /= iteration_altopt
        time_complexity_p /= iteration_altopt

        print("\n****************************************************************************************************************************")
        print(f"Alternating Optimization -> Number of Steps: {iteration_altopt}, p_sol_tot: {np.sum(p_sol)}, gamma_sol_norm: {np.linalg.norm(gamma_sol)}, gamma_sol_norm_Q: {np.linalg.norm(gamma_sol_Q)}, SSR_opt: {ssr_sol}, SSR_opt_Q: {ssr_sol_Q}, SEE_opt: {see_sol}, SEE_opt_Q: {see_sol_Q}, iteration_altopt: {iteration_altopt}, iteration_p: {iteration_p}, iteration_gamma: {iteration_gamma}, time_complexity_altopt: {time_complexity_altopt}, time_complexity_p: {time_complexity_p}, time_complexity_gamma: {time_complexity_gamma}")
        print("****************************************************************************************************************************\n")

        return p_sol, gamma_sol, gamma_sol_Q, ssr_sol, ssr_sol_Q, see_sol, see_sol_Q, iteration_altopt, iteration_p, iteration_gamma, time_complexity_altopt, time_complexity_p, time_complexity_gamma
