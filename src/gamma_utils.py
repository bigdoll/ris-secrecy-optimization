import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm
from utils import Utils
<<<<<<< HEAD
from typing import Union

class GammaUtils:
    def __init__(self, H, G_B, G_E, sigma_sq, sigma_RIS_sq, sigma_e_sq, mu, Pc, scsi_bool=0, utils_cls=Utils):
=======

class GammaUtils:
    def __init__(self, H, G_B, G_E, sigma_sq, sigma_RIS_sq, sigma_g_sq, mu, Pc, scsi_bool=1, utils_cls=Utils):
>>>>>>> origin/main
        """
        Initialize the GammaUtils class with given parameters.

        Parameters:
        - H: Channel matrix from UEs to RIS.
        - G_B: Channel matrix from RIS to Bob.
        - G_E: Channel matrix from RIS to Eve.
        - sigma_sq: Noise power variance at the receiver.
        - sigma_RIS_sq: Noise power variance at RIS.
        - sigma_g_sq: Noise variance for Eve's channel estimation error.
        - mu: Amplifier inefficiency factor.
        - Pc: Static power consumption.
        - scsi_bool: Boolean indicating whether to consider channel state information (CSI).
        - utils_cls: Reference to the Utils class for dependency injection.
        """
        self.H = H
        self.G_B = G_B
        self.G_E = G_E
        self.sigma_sq = sigma_sq
        self.sigma_RIS_sq = sigma_RIS_sq
<<<<<<< HEAD
        self.sigma_e_sq = sigma_e_sq
=======
        self.sigma_g_sq = sigma_g_sq
>>>>>>> origin/main
        self.mu = mu
        self.Pc = Pc
        self.scsi_bool = scsi_bool
        self.utils_cls = utils_cls
<<<<<<< HEAD
        self.K = H.shape[1]  # number of users
        self.N = H.shape[0]  # number of RIS elements
        # Precompute Eve covariance
        self.R_E = G_E @ G_E.conj().T +  scsi_bool * sigma_e_sq * np.eye(self.N)  # to be set externally: 
        # e.g., self.R_E = gE_hat @ gE_hat.conj().T + sigma_g_sq * np.eye(self.N)
=======
>>>>>>> origin/main
    
    def compute_R(self, p):
        """
        Wrapper method to call the static method compute_R from Utils.
        
        Parameters:
        - p: Power allocation vector for UEs.

        Returns:
        - R: Computed R matrix.
        """
        return self.utils_cls.compute_R(self.H, p, self.sigma_RIS_sq)
    
    def sinr_active_Bob(self, C, gamma_bar, p):
        """
        Wrapper method to call the static method sinr_active_Bob from Utils.
        
        Parameters:
        - C: Linear MMSE receive filters.
        - gamma_bar: Current reflection coefficients.
        - p: Power allocation vector for UEs.

        Returns:
        - sinr_a: SINR for active Bob.
        """
        return self.utils_cls.sinr_active_Bob(C, self.G_B, self.H, gamma_bar, p, self.sigma_sq, self.sigma_RIS_sq)
    
    def LMMSE_receiver_active_Bob(self, gamma_bar, p):
        """
        Wrapper method to call the static method LMMSE_receiver_active_Bob from Utils.
        
        Parameters:
        - gamma_bar: Current reflection coefficients.
        - p: Power allocation vector for UEs.

        Returns:
        - C: Linear MMSE receive filters.
        """
        return self.utils_cls.LMMSE_receiver_active_Bob(self.G_B, self.H, gamma_bar, p, self.sigma_sq, self.sigma_RIS_sq)
    
    def compute_Pc_eq_gamma(self, p):
        """
        Compute the equivalent power consumption considering gamma.

        Parameters:
        - p: Power allocation vector for UEs.

        Returns:
        - Pc_eq: Equivalent power consumption.
        """
        K = len(p)
        Pc_eq = 0

        R = self.compute_R(p)

        for k in range(K):
            Pc_eq += p[k] * self.mu

        Pc_eq += self.Pc - np.real(np.trace(R))
        return Pc_eq

    def parameters_active_Bob(self, C, gamma_bar, p):
        """
        Compute parameters required for Bob's SINR and data rate calculations.

        Parameters:
        - C: Linear MMSE receive filters.
        - gamma_bar: Current reflection coefficients.
        - p: Power allocation vector for UEs.

        Returns:
        - A_bar, B_bar, D_bar, E_bar, F_bar: Parameters for optimization.
        """
        epsilon = np.finfo(float).eps
        K = p.shape[0]
        NR = self.G_B.shape[0]

        sinr_a = self.sinr_active_Bob(C, gamma_bar, p)
        A_bar = np.log2(1 + sinr_a)
        B_bar = sinr_a

        D_bar = np.zeros_like(p)
        for k in range(K):
            ck = C[:, k].reshape(NR, 1)
            hk = self.H[:, k]
            Hk = np.diag(hk)
            Ak = self.G_B @ Hk
            denom_k = np.sum(np.abs(ck.conj().T @ Ak @ gamma_bar))
            D_bar[k] = 2 / np.where(denom_k > epsilon, denom_k, epsilon)

        E_bar = np.zeros_like(p)
        for k in range(K):
            ck = C[:, k].reshape(NR, 1)
            pakm_sum = 0
            for m in range(K):
                hm = self.H[:, m]
                Hm = np.diag(hm)
                Am = self.G_B @ Hm
                akm = np.sum(np.abs(ck.conj().T @ Am @ gamma_bar) ** 2)
                pakm_sum += p[m] * akm
            uk = self.G_B.conj().T @ ck
            Uk_tilt = np.diagflat(np.abs(uk) ** 2)
            noise_k = self.sigma_sq * np.linalg.norm(ck) ** 2 + self.sigma_RIS_sq * np.linalg.norm(sqrtm(Uk_tilt) @ gamma_bar)**2
            denom_k = noise_k + pakm_sum
            E_bar[k] = 1 / np.where(denom_k > epsilon, denom_k, epsilon)

        F_bar = np.zeros_like(p)
        for k in range(K):
            ck = C[:, k].reshape(NR, 1)
            F_bar[k] = E_bar[k] * self.sigma_sq * np.linalg.norm(ck) ** 2 + 1

        return A_bar, B_bar, D_bar, E_bar, F_bar

    def parameters_active_Eve(self, gamma_bar, p):
        """
        Compute parameters required for Eve's SINR and data rate calculations.

        Parameters:
        - gamma_bar: Current reflection coefficients.
        - p: Power allocation vector for UEs.

        Returns:
        - A_bar, B_bar, D_bar, E_bar, F_bar, L_bar, Q_bar, V_bar: Parameters for optimization.
        """
        epsilon = np.finfo(float).eps
        K = p.shape[0]
        N = self.H.shape[0]

<<<<<<< HEAD
        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_e_sq * np.eye(N)
=======
        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_g_sq * np.eye(N)
>>>>>>> origin/main

        xE_bar = np.zeros_like(p)
        for k in range(K):
            hk = self.H[:, k]
            Hk = np.diag(hk)
            xE_bar[k] = p[k] * np.linalg.norm(sqrtm(RE) @ Hk @ gamma_bar)**2

        yE_bar = np.zeros_like(p)
        for k in range(K):
            interf_m = 0
            for m in range(K):
                if m != k:
                    hm = self.H[:, m]
                    Hm = np.diag(hm)
                    interf_m += p[m] * np.linalg.norm(sqrtm(RE) @ Hm @ gamma_bar) ** 2
            yE_bar[k] = interf_m + self.sigma_RIS_sq * np.linalg.norm(sqrtm(RE) @ gamma_bar)**2

        A_bar = np.log2(1 + yE_bar / self.sigma_sq)
        B_bar = yE_bar / self.sigma_sq
        yE_bar_sqrt = np.sqrt(yE_bar)
        D_bar = 2 / np.where(yE_bar_sqrt > epsilon, yE_bar_sqrt, epsilon)

        E_bar = np.zeros_like(p)
        for k in range(K):
            denom_k = self.sigma_sq + yE_bar[k]
            E_bar[k] = 1 / np.where(denom_k > epsilon, denom_k, epsilon)

        F_bar = np.zeros_like(p)
        for k in range(K):
            F_bar[k] = E_bar[k] * self.sigma_sq + 1

        Q_bar = 1 / (self.sigma_sq + xE_bar + yE_bar)
        L_bar = np.log2(self.sigma_sq * Q_bar)
        V_bar = Q_bar * (xE_bar + yE_bar)

        return A_bar, B_bar, D_bar, E_bar, F_bar, L_bar, Q_bar, V_bar

    def SR_active_concave_gamma_Bob(self, gamma, gamma_bar, p, cvx_bool):
        """
        Compute the concave approximation of the secrecy rate for Bob.

        Parameters:
        - gamma: Current reflection coefficients.
        - gamma_bar: Previous reflection coefficients.
        - p: Power allocation vector for UEs.
        - cvx_bool: Boolean indicating whether to use convex approximation.

        Returns:
        - sr_concave: Concave approximation of the secrecy rate.
        """
        epsilon = np.finfo(float).eps
        K = p.shape[0]
        NR = self.G_B.shape[0]
        sr_concave = 0

        C = self.LMMSE_receiver_active_Bob(gamma_bar, p)
        A_bar, B_bar, D_bar, E_bar, F_bar = self.parameters_active_Bob(C, gamma_bar, p)

        for k in range(K):
            ck = C[:, k].reshape(NR, 1)
            hk = self.H[:, k]
            Hk = np.diag(hk)
            Ak = self.G_B @ Hk
            Yp = 0

            for m in range(K):
                hm = self.H[:, m]
                Hm = np.diag(hm)
                Am = self.G_B @ Hm
                pm = p[m]
                Yp += pm * cp.sum(cp.abs(ck.conj().T @ Am @ gamma) ** 2)

            uk = self.G_B.conj().T @ ck
            Uk_tilt = np.diagflat(np.abs(uk) ** 2)
            Zk = np.sum(np.abs(ck.conj().T @ Ak @ gamma_bar))
            deriv = Ak.conj().T @ (ck @  ck.conj().T) @ Ak @ gamma_bar / max(Zk, epsilon)

            sr_concave += A_bar[k] * cvx_bool + (B_bar[k] * D_bar[k]) * (np.sum(np.abs(ck.conj().T @ Ak @ gamma_bar)) * cvx_bool + cp.real(cp.sum(cp.matmul(deriv.conj().T, gamma - gamma_bar * cvx_bool)))) - (B_bar[k] * F_bar[k]) * cvx_bool - (B_bar[k] * E_bar[k]) * (self.sigma_RIS_sq * cp.real(cp.sum(cp.quad_form(gamma, Uk_tilt))) + Yp)

        return sr_concave

    def SR_active_concave_gamma_Eve(self, gamma, gamma_bar, p, cvx_bool):
        """
        Compute the concave approximation of the secrecy rate for Eve.

        Parameters:
        - gamma: Current reflection coefficients.
        - gamma_bar: Previous reflection coefficients.
        - p: Power allocation vector for UEs.
        - cvx_bool: Boolean indicating whether to use convex approximation.

        Returns:
        - sr_concave: Concave approximation of the secrecy rate.
        """
        epsilon = np.finfo(float).eps
        K = p.shape[0]
        N = self.H.shape[0]
<<<<<<< HEAD
        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_e_sq * np.eye(N)
=======
        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_g_sq * np.eye(N)
>>>>>>> origin/main
        sr_concave = 0

        A_bar, B_bar, D_bar, E_bar, F_bar, L_bar, Q_bar, V_bar = self.parameters_active_Eve(gamma_bar, p)

        for k in range(K):
<<<<<<< HEAD
            xE = 0 #p[k] * cp.real(cp.sum(cp.quad_form(gamma, self.H[:, k].conj().T @ RE @ self.H[:, k])))
=======
            xE = p[k] * cp.real(cp.sum(cp.quad_form(gamma, self.H[:, k].conj().T @ RE @ self.H[:, k])))
>>>>>>> origin/main
            yE = 0
            interf_m = 0
            interf_m_bar = 0
            interf_m_grad = 0
            hk = self.H[:, k]
            Hk = np.diag(hk)
            Zk = Hk.conj().T @ RE @ Hk

            for m in range(K):
                if m != k:
                    hm = self.H[:, m]
                    Hm = np.diag(hm)
                    Zm = Hm.conj().T @ RE @ Hm
                    interf_m += p[m] * cp.real(cp.sum(cp.quad_form(gamma, Zm)))
                    interf_m_bar += p[m] * np.linalg.norm(sqrtm(RE) @ Hm @ gamma_bar)**2
                    interf_m_grad += p[m] * Zm @ gamma_bar

<<<<<<< HEAD
            xE = p[k] * cp.real(cp.sum(cp.quad_form(gamma, Zk)))
=======
>>>>>>> origin/main
            yE = interf_m + self.sigma_RIS_sq * cp.real(cp.sum(cp.quad_form(gamma, RE)))
            yE_bar = interf_m_bar + self.sigma_RIS_sq * np.linalg.norm(sqrtm(RE) @ gamma_bar)**2
            yE_bar_sqrt = np.sqrt(yE_bar)
            grad_yE_bar = interf_m_grad + self.sigma_RIS_sq * RE @ gamma_bar
            grad_yE_sqrt_bar = grad_yE_bar / np.where(yE_bar_sqrt > epsilon, yE_bar_sqrt, epsilon)

            sr_concave += A_bar[k] * cvx_bool + (B_bar[k] * D_bar[k]) * (yE_bar_sqrt * cvx_bool + cp.real(cp.sum(cp.matmul(grad_yE_sqrt_bar.conj().T, (gamma - gamma_bar * cvx_bool))))) - (B_bar[k] * F_bar[k]) * cvx_bool - (B_bar[k] * E_bar[k]) * yE + (L_bar[k] + V_bar[k]/np.log(2)) * cvx_bool - (Q_bar[k]/np.log(2)) * (xE + yE)

        return sr_concave
<<<<<<< HEAD
    
    def SR_concave_gamma_Bob_X(self,
                           X: Union[np.ndarray, cp.Expression],
                           p: np.ndarray,
                           X0: np.ndarray = None,
                           cvx_bool: bool = False) -> Union[float, cp.Expression]:
        """
        Surrogate lower‐bound for Bob's rate:
        sum_{k} [ log2 det(W_k + p_k A_k X A_k^H)  -  log2 det(W_k) ]
        where 
        W_k(X) = sigma^2 I  +  sigma_RIS^2 G_B diag(X) G_B^H
                + sum_{m≠k} p_m A_m X A_m^H.
        
        We now rewrite it “noise‐normalized” inside this function:
        - Replace every p_m by p_m' = p_m / sigma^2
        - Replace sigma_RIS^2 by sigma_RIS'^2 = sigma_RIS^2 / sigma^2
        - Replace “sigma^2 I” by “I”
        - Add a tiny +ε·I to every log_det argument.
        
        Outside this function, you keep your original constraints and Dinkelbach power term 
        unchanged, so the global maximizer gamma* remains exactly the same.
        """
        K   = self.K
        N_B = self.G_B.shape[0]
        log2 = np.log(2)
        eps  = 0   # small regularizer 1e-6

        # 1) Compute the “scaled” quantities once:
        sigma0_sq            = self.sigma_sq               # original noise variance
        p_scaled             = p / sigma0_sq               # p_m' = p_m / sigma^2
        sigma_RIS_sq_scaled  = self.sigma_RIS_sq / sigma0_sq
        G_B_scaled           = self.G_B                    # unchanged
        I_B                  = np.eye(N_B)

        if cvx_bool:
            if X0 is None:
                raise ValueError("X0 required for SCA surrogate.")

            # 2) Build the “noise‐normalized” W(X):
            #    W_scaled(X) = I_B + sigma_RIS_sq_scaled * G_B diag(X) G_B^H
            W = I_B + sigma_RIS_sq_scaled * (G_B_scaled @ cp.diag(cp.diag(X)) @ G_B_scaled.conj().T)

            # 3) Precompute A_m X A_m^H at current X for all m
            A_X_AT = [
                (G_B_scaled @ np.diag(self.H[:, m])) @ X @
                (G_B_scaled @ np.diag(self.H[:, m])).conj().T
                for m in range(K)
            ]
            S_total = sum(p_scaled[m] * A_X_AT[m] for m in range(K))

            term_exact = 0
            lin_sum    = 0
            const_sum  = 0

            # 4) Build everything at X0 for the linearization
            W0 = I_B + sigma_RIS_sq_scaled * (G_B_scaled @ np.diag(np.diag(X0)) @ G_B_scaled.conj().T)
            A_X0_AT = [
                (G_B_scaled @ np.diag(self.H[:, m])) @ X0 @
                (G_B_scaled @ np.diag(self.H[:, m])).conj().T
                for m in range(K)
            ]
            S0_total = sum(p_scaled[m] * A_X0_AT[m] for m in range(K))

            for k in range(K):
                # 5) interference‐plus‐noise for user k (noise‐normalized):
                Wk  = W   + (S_total - p_scaled[k] * A_X_AT[k])
                Wk0 = W0  + (S0_total - p_scaled[k] * A_X0_AT[k])
                S_k = p_scaled[k] * A_X_AT[k]

                # 6) EXACT concave term:  log_det(Wk + p_scaled[k] A_k X A_k^H + ε I)
                Wk_plus_Sk = Wk + S_k + eps * I_B
                term_exact += cp.log_det(Wk_plus_Sk) / log2

                # 7) Build the numeric gradient of logdet(Wk) at X0
                #    We need Wk0 to be a numpy array when X0 is numeric
                Wk0_np = Wk0 if isinstance(Wk0, np.ndarray) else np.array(Wk0)
                # Add eps·I here as well to keep it strictly PD
                Wk0_inv = np.linalg.inv(Wk0_np + eps * I_B)

                # 8) Interference gradient: sum_{m≠k} p_scaled[m] * A_m^H Wk0_inv A_m
                grad_interf = sum(
                    p_scaled[m] * (G_B_scaled @ np.diag(self.H[:, m])).conj().T
                                @ Wk0_inv
                                @ (G_B_scaled @ np.diag(self.H[:, m]))
                    for m in range(K) if m != k
                )

                # 9) RIS‐noise gradient (only diag‐entries)
                GtWk0    = G_B_scaled.conj().T @ Wk0_inv @ G_B_scaled
                grad_noise = sigma_RIS_sq_scaled * np.diag(np.diag(GtWk0))

                grad_k = grad_interf + grad_noise

                # 10) Linearize - log2 det(Wk):
                const_sum += -np.real(np.log2(np.linalg.det(Wk0_np + eps * I_B))) \
                            + (1 / log2) * np.real(np.trace(grad_k @ X0))
                lin_sum   += -(1 / log2) * cp.real(cp.trace(grad_k @ X))

            return term_exact + const_sum + lin_sum

        else:
            # Numeric (cvx_bool=False) evaluation of Bob's rate at a given X
            W = I_B + sigma_RIS_sq_scaled * (G_B_scaled @ np.diag(np.diag(X)) @ G_B_scaled.conj().T)
            sr = 0.0

            for k in range(K):
                A_k = G_B_scaled @ np.diag(self.H[:, k])
                S_k = p_scaled[k] * A_k @ X @ A_k.conj().T

                # interference‐plus‐noise for user k
                Wk = W.copy()
                for m in range(K):
                    if m != k:
                        A_m = G_B_scaled @ np.diag(self.H[:, m])
                        Wk += p_scaled[m] * A_m @ X @ A_m.conj().T

                # Add eps·I to both det‐arguments:
                val1 = np.log2(np.linalg.det(Wk + S_k + eps * I_B))
                val2 = np.log2(np.linalg.det(Wk + eps * I_B))
                sr += val1 - val2

            return np.real(sr)



    
    def SR_concave_gamma_Eve_X(self,
                           X: Union[np.ndarray, cp.Expression],
                           p: np.ndarray,
                           X0: np.ndarray = None,
                           cvx_bool: bool = False) -> Union[float, cp.Expression]:
        """
        Surrogate for Eve’s rate:
        sum_k  log2(1 + num_k / den_k),
        where 
        num_k = p_k * tr(M_k X),
        den_k = sum_{m≠k} p_m tr(M_m X) + sigma_RIS^2 tr(R_E X) + sigma^2.
        
        We rewrite “noise‐normalized” inside this function:
        - p_k' = p_k / sigma^2,
        - sigma_RIS'^2 = sigma_RIS^2 / sigma^2,
        - R_E' = R_E / sigma^2,
        - and set “sigma^2 → 1” in every denominator.
        - Add a tiny ε to the log‐argument so it never sees zero.
        
        Outside this function, you keep all constraints D, R, etc. exactly as before.
        """
        if self.R_E is None:
            raise ValueError("R_E must be set before calling this.")
        ln2 = np.log(2)
        eps = 0 # small Regularizer 1e-6

        # 1) Scaled versions:
        sigma0_sq           = self.sigma_sq
        p_scaled            = p / sigma0_sq
        sigma_RIS_sq_scaled = self.sigma_RIS_sq / sigma0_sq
        R_E_scaled          = self.R_E / sigma0_sq

        # Precompute M_list with scaled R_E
        M_list = [
            np.diag(self.H[:, k]).conj().T @ R_E_scaled @ np.diag(self.H[:, k])
            for k in range(self.K)
        ]
        D_base = sigma_RIS_sq_scaled * R_E_scaled

        if cvx_bool:
            if X0 is None:
                raise ValueError("X0 required for SCA surrogate.")

            expr = 0
            const = 0

            # 2) Precompute numeric traces at X0
            tr_M_X0 = [np.real(np.trace(M_list[k] @ X0)) for k in range(self.K)]
            tr_R_X0 = np.real(np.trace(R_E_scaled @ X0))

            for k in range(self.K):
                # Build D_k = sigma_RIS_sq_scaled R_E_scaled + sum_{m≠k} p_scaled[m] M_list[m]
                D_k = D_base + sum(p_scaled[m] * M_list[m]
                                for m in range(self.K) if m != k)

                # Numeric interference at X0
                inter_X0 = sum(p_scaled[m] * np.real(np.trace(M_list[m] @ X0))
                            for m in range(self.K) if m != k)
                den0     = inter_X0 + sigma_RIS_sq_scaled * tr_R_X0 + 1   # “noise = 1”
                num0     = p_scaled[k] * tr_M_X0[k]
                f0       = num0 + den0

                # Gradient of log2(f_k) at X0
                grad_h = (1 / ln2) * (1 / f0) * (p_scaled[k] * M_list[k] + D_k)

                # Surrogate for log2(f_k):
                h0    = np.log2(f0)
                const += h0 - np.real(np.trace(grad_h @ X0))
                expr  += cp.real(cp.trace(grad_h @ X))

                # Exact -log2( den_expr ), with “+ε” so it never hits log(0)
                den_expr = cp.real(cp.trace(D_k @ X)) + 1  # “noise = 1”
                expr     += - (1 / ln2) * cp.log(den_expr + eps)

            return expr + const

        else:
            total = 0.0
            tr_R_X = np.real(np.trace(R_E_scaled @ X))

            for k in range(self.K):
                M_k = M_list[k]
                num = p_scaled[k] * np.real(np.trace(M_k @ X))

                inter = sum(p_scaled[m] * np.real(np.trace(M_list[m] @ X))
                            for m in range(self.K) if m != k)
                den   = inter + sigma_RIS_sq_scaled * tr_R_X + 1  # “noise = 1”

                total += np.real(np.log2(1 + num / (den + eps)))

            return total



    # def SR_concave_gamma_Bob_X(self,
    #                        X: Union[np.ndarray, cp.Expression],
    #                        p: np.ndarray,
    #                        X0: np.ndarray = None,
    #                        cvx_bool: bool = False) -> Union[float, cp.Expression]:
    #     """
    #     Surrogate lower‐bound for Bob's rate:
    #     sum_k [ log2 det(W_k + p_k A_k X A_k^H) - log2 det(W_k) ]
    #     where W_k(X) = W(X) + sum_{m!=k} p_m A_m X A_m^H.
    #     We keep the logdet(+p_k A_k X A_k^H) term exact (concave)
    #     and linearize only the -logdet(W_k) around X0.
    #     """
    #     K = self.K
    #     N_B = self.G_B.shape[0]
    #     log2 = np.log(2)

    #     # Base noise+RIS‐noise term W(X)
    #     if cvx_bool:
    #         if X0 is None:
    #             raise ValueError("X0 required for SCA surrogate.")

    #         I = np.eye(N_B)
    #         W = self.sigma_sq * I \
    #             + self.sigma_RIS_sq * self.G_B @ cp.diag(cp.diag(X)) @ self.G_B.conj().T

    #         # Precompute per‐user A_m @ X @ A_m^H
    #         A_X_AT = [
    #             (self.G_B @ np.diag(self.H[:, m])) @ X @
    #             (self.G_B @ np.diag(self.H[:, m])).conj().T
    #             for m in range(K)
    #         ]
    #         S_total = sum(p[m] * A_X_AT[m] for m in range(K))

    #         term_exact = 0
    #         lin_sum   = 0
    #         const_sum = 0

    #         # Precompute at X0 for surrogate
    #         W0 = self.sigma_sq * np.eye(N_B) \
    #             + self.sigma_RIS_sq * self.G_B @ np.diag(np.diag(X0)) @ self.G_B.conj().T
    #         A_X0_AT = [
    #             (self.G_B @ np.diag(self.H[:, m])) @ X0 @
    #             (self.G_B @ np.diag(self.H[:, m])).conj().T
    #             for m in range(K)
    #         ]
    #         S0_total = sum(p[m] * A_X0_AT[m] for m in range(K))

    #         for k in range(K):
    #             # interference‐plus‐noise for user k
    #             Wk     = W + (S_total - p[k] * A_X_AT[k])
    #             Wk0    = W0 + (S0_total - p[k] * A_X0_AT[k])
    #             S_k    = p[k] * A_X_AT[k]

    #             # exact concave term
    #             term_exact += cp.log_det(Wk + S_k) / log2

    #             # build gradient of logdet(Wk) at X0
    #             Wk0_inv = np.linalg.inv(Wk0)

    #             # interference gradient: sum_{m!=k} p_m A_m^H Wk0_inv A_m
    #             grad_interf = sum(
    #                 p[m] * (self.G_B @ np.diag(self.H[:, m])).conj().T
    #                         @ Wk0_inv
    #                         @ (self.G_B @ np.diag(self.H[:, m]))
    #                 for m in range(K) if m != k
    #             )

    #             # RIS‐noise gradient: only on diag entries of X
    #             GtWk0 = self.G_B.conj().T @ Wk0_inv @ self.G_B
    #             grad_noise = self.sigma_RIS_sq * np.diag(np.diag(GtWk0))

    #             grad_k = grad_interf + grad_noise

    #             # surrogate: linearize -log2 det(Wk)
    #             const_sum += -np.real(np.log2(np.linalg.det(Wk0))) \
    #                         + (1/log2) * np.real(np.trace(grad_k @ X0))
    #             lin_sum   += -(1/log2) * cp.real(cp.trace(grad_k @ X))

    #         return term_exact + const_sum + lin_sum

    #     else:
    #         # numeric evaluation
    #         I = np.eye(N_B)
    #         W = self.sigma_sq * I + self.sigma_RIS_sq * self.G_B @ np.diag(np.diag(X)) @ self.G_B.conj().T
    #         sr = 0.0

    #         for k in range(self.K):
    #             A_k = self.G_B @ np.diag(self.H[:, k])
    #             S_k = p[k] * A_k @ X @ A_k.conj().T

    #             # build W_k = W + interference from m != k
    #             Wk = W.copy()
    #             for m in range(self.K):
    #                 if m != k:
    #                     A_m = self.G_B @ np.diag(self.H[:, m])
    #                     Wk += p[m] * A_m @ X @ A_m.conj().T

    #             # sum log‐det difference
    #             val1 = np.log2(np.linalg.det(Wk + S_k)) # np.linalg.slogdet(Wk + S_k)[1] / log2
    #             val2 = np.log2(np.linalg.det(Wk)) # np.linalg.slogdet(Wk)[1]     / log2
    #             sr += val1 - val2

    #         return np.real(sr)
        
    
    # def SR_concave_gamma_Eve_X(self, X, p, X0=None, cvx_bool=False):
    #     if self.R_E is None:
    #         raise ValueError("R_E must be set before calling this.")
    #     ln2 = np.log(2)

    #     # Precompute M_k = H_k^H R_E H_k for all k
    #     M_list = [
    #         np.diag(self.H[:,k]).conj().T @ self.R_E @ np.diag(self.H[:,k])
    #         for k in range(self.K)
    #     ]
    #     D_base = self.sigma_RIS_sq * self.R_E

    #     if cvx_bool:
    #         if X0 is None:
    #             raise ValueError("X0 required for SCA surrogate.")
    #         expr = 0
    #         const = 0

    #         # Precompute traces at X0
    #         tr_M_X0 = [np.real(np.trace(M_list[k] @ X0)) for k in range(self.K)]
    #         tr_R_X0 = np.real(np.trace(self.R_E @ X0))

    #         for k in range(self.K):
    #             # build D_k = σ_RIS² R_E + sum_{m≠k} p[m] M_list[m]
    #             D_k = D_base + sum(p[m] * M_list[m]
    #                             for m in range(self.K) if m != k)

    #             num0 = p[k] * tr_M_X0[k]
    #             den0 = np.real(np.trace(D_k @ X0)) + self.sigma_sq
    #             f0   = num0 + den0

    #             # gradient of log2(f) at X0
    #             grad_h = (1/ln2) * (1/f0) * (p[k]*M_list[k] + D_k)

    #             # surrogate of log2(f)
    #             h0 = np.log2(f0)
    #             const += h0 - np.real(np.trace(grad_h @ X0))
    #             expr  += cp.real(cp.trace(grad_h @ X))

    #             # exact -log2(den) term
    #             den_expr = cp.real(cp.trace(D_k @ X)) + self.sigma_sq
    #             expr     += -cp.log(den_expr)/ln2

    #         return expr + const

    #     else:
    #         total = 0.0
    #         tr_R_X = np.real(np.trace(self.R_E @ X))
    #         for k in range(self.K):
    #             M_k = M_list[k]
    #             num = p[k] * np.real(np.trace(M_k @ X))

    #             inter = sum(p[m] * np.real(np.trace(M_list[m] @ X))
    #                         for m in range(self.K) if m != k)
    #             den = inter + self.sigma_RIS_sq * tr_R_X + self.sigma_sq

    #             total += np.real(np.log2(1 + num/den))
    #         return total

    # def SR_concave_gamma_Eve_X(self, X, p, X0=None, cvx_bool=False):
    #     if self.R_E is None:
    #         raise ValueError("R_E must be set before calling this.")
    #     ln2 = np.log(2)

    #     # Precompute M_k = H_k^H R_E H_k for all k
    #     M_list = [
    #         np.diag(self.H[:,k]).conj().T @ self.R_E @ np.diag(self.H[:,k])
    #         for k in range(self.K)
    #     ]
    #     D_base = self.sigma_RIS_sq * self.R_E

    #     if cvx_bool:
    #         if X0 is None:
    #             raise ValueError("X0 required for SCA surrogate.")
    #         expr = 0
    #         const = 0

    #         # Precompute traces at X0
    #         tr_M_X0 = [np.real(np.trace(M_list[k] @ X0)) for k in range(self.K)]
    #         tr_R_X0 = np.real(np.trace(self.R_E @ X0))

    #         for k in range(self.K):
    #             # build D_k = σ_RIS² R_E + sum_{m≠k} p[m] M_list[m]
    #             D_k = D_base + sum(p[m] * M_list[m]
    #                             for m in range(self.K) if m != k)

    #             num0 = p[k] * tr_M_X0[k]
    #             den0 = np.real(np.trace(D_k @ X0)) + self.sigma_e_sq
    #             f0   = num0 + den0

    #             # gradient of log2(f) at X0
    #             grad_h = (1/ln2) * (1/f0) * (p[k]*M_list[k] + D_k)

    #             # surrogate of log2(f)
    #             h0 = np.log2(f0)
    #             const += h0 - np.real(np.trace(grad_h @ X0))
    #             expr  += cp.real(cp.trace(grad_h @ X))

    #             # exact -log2(den) term
    #             den_expr = cp.real(cp.trace(D_k @ X)) + self.sigma_e_sq
    #             expr     += -cp.log(den_expr)/ln2

    #         return expr + const

    #     else:
    #         total = 0.0
    #         tr_R_X = np.real(np.trace(self.R_E @ X))
    #         for k in range(self.K):
    #             M_k = M_list[k]
    #             num = p[k] * np.real(np.trace(M_k @ X))

    #             inter = sum(p[m] * np.real(np.trace(M_list[m] @ X))
    #                         for m in range(self.K) if m != k)
    #             den = inter + self.sigma_RIS_sq * tr_R_X + self.sigma_e_sq

    #             total += np.log2(1 + num/den)
    #         return total


    # def SR_concave_gamma_Bob_X(self, X, p, X0=None, cvx_bool=False):
    #     """
    #     Surrogate lower-bound for Bob rate: logdet(W+S) - logdet(W)
    #     Linearize only the -logdet(W) term around X0.
    #     """
    #     N_B = self.G_B.shape[0] # self.N
    #     # Define W(X) and S(X)
    #     if cvx_bool:
    #         if X0 is None:
    #             raise ValueError("X0 required for SCA surrogate.")
    #         I = np.eye(N_B)
    #         # W(X) in CVX
    #         W = self.sigma_sq * I + self.sigma_RIS_sq * self.G_B @ cp.diag(cp.diag(X)) @ self.G_B.conj().T
    #         # S(X)
    #         S = 0
    #         for k in range(self.K):
    #             A = self.G_B @ np.diag(self.H[:, k])
    #             S += p[k] * A @ X @ A.conj().T
    #         # First term concave
    #         term1 = cp.log_det(W + S) / np.log(2)
    #         # Compute gradient of logdet(W0)
    #         # Numeric evaluation at X0
    #         W0 = self.sigma_sq * np.eye(N_B) + self.sigma_RIS_sq * self.G_B @ np.diag(np.diag(X0)) @ self.G_B.conj().T
    #         W0_inv = np.linalg.inv(W0)
    #         # d/dX logdet(W) = sigma_RIS_sq * diag( G_B^H W0_inv G_B )
    #         gradW = self.sigma_RIS_sq * np.diag(np.diag(self.G_B.conj().T @ W0_inv @ self.G_B))
    #         # Surrogate for -logdet(W)
    #         const = - np.real((np.log2(np.linalg.det(W0)))) + (1/np.log(2)) * np.real(np.trace(gradW @ X0))
    #         lin = - (1/np.log(2)) * cp.real(cp.trace(gradW @ X))
    #         return term1 + const + lin
    #     else:
    #         W = self.sigma_sq * np.eye(N_B) + self.sigma_RIS_sq * self.G_B @ np.diag(np.diag(X)) @ self.G_B.conj().T
    #         S = sum(p[k] * (self.G_B @ np.diag(self.H[:, k]) @ X @ np.diag(self.H[:, k]).conj().T @ self.G_B.conj().T)
    #                 for k in range(self.K))
    #         return np.real((np.log2(np.linalg.det(W + S)) - np.log2(np.linalg.det(W))))

    # def SR_concave_gamma_Eve_X(self, X, p, X0=None, cvx_bool=False):
    #     """
    #     Surrogate lower-bound for Eve's approximate sum-rate:
    #     R_E(X) = sum_k [log2(den_k + num_k) - log2(den_k)]
    #     Linearize only the concave log2(den+num) term around X0, keep -log2(den) exact.
    #     Requires self.R_E set.
    #     """
    #     if self.R_E is None:
    #         raise ValueError("R_E must be set before calling this.")
    #     ln2 = np.log(2)
    #     # Precompute common D_k for each k
    #     if cvx_bool:
    #         if X0 is None:
    #             raise ValueError("X0 required for SCA surrogate.")
    #         expr = 0
    #         const_sum = 0.0
    #         # Loop over users
    #         for k in range(self.K):
    #             # Matrices
    #             Hk = np.diag(self.H[:, k])
    #             M_k = Hk.conj().T @ self.R_E @ Hk
    #             D_k = self.sigma_RIS_sq * self.R_E
    #             for m in range(self.K):
    #                 if m == k:
    #                     continue
    #                 Hm = np.diag(self.H[:, m])
    #                 D_k += p[m] * (Hm.conj().T @ self.R_E @ Hm)
    #             # Numeric at X0
    #             num0 = p[k] * np.real(np.trace(M_k @ X0))
    #             den0 = np.real(np.trace(D_k @ X0)) + self.sigma_e_sq
    #             f0 = den0 + num0
    #             # Gradient of log2(f) at X0
    #             grad_h = (1.0/ln2) * (1.0/f0) * (p[k] * M_k + D_k)
    #             # Constant term for h0 - grad_h @ X0
    #             h0 = np.real(np.log2(f0))
    #             const_sum += (h0 - np.real(np.trace(grad_h @ X0)))
    #             # Accumulate linearized h part
    #             expr += cp.real(cp.trace(grad_h @ X))
    #             # Subtract exact -log2(den) term
    #             den_expr = cp.real(cp.trace(D_k @ X)) + self.sigma_e_sq
    #             expr += - cp.log(den_expr) / ln2
    #         return expr + const_sum
    #     else:
    #         total = 0.0
    #         for k in range(self.K):
    #             Hk = np.diag(self.H[:, k])
    #             M_k = Hk.conj().T @ self.R_E @ Hk
    #             num = p[k] * np.real(np.trace(M_k @ X))
    #             inter = 0.0
    #             for m in range(self.K):
    #                 if m == k:
    #                     continue
    #                 Hm = np.diag(self.H[:, m])
    #                 inter += p[m] * np.real(np.trace(Hm.conj().T @ self.R_E @ Hm @ X))
    #             den = inter + self.sigma_RIS_sq * np.real(np.trace(self.R_E @ X)) + self.sigma_e_sq
    #             total += np.log2(1 + num/den)
    #         return total



=======
>>>>>>> origin/main

    def compute_grad_gamma(self, C, gamma, p):
        """
        Compute the gradient of the objective function with respect to gamma.

        Parameters:
        - C: Linear MMSE receive filters.
        - gamma: Current reflection coefficients.
        - p: Power allocation vector for UEs.

        Returns:
        - grad_gamma: Gradient of the objective function with respect to gamma.
        """
        epsilon = np.finfo(float).eps
        K = p.shape[0]

        xB, yB, xE, yE = self.params(C, gamma, p)
        grad_xB, grad_yB, grad_xE, grad_yE = self.grad_params(C, gamma, p)

        grad_f1 = 0
        grad_f2 = 0

        for k in range(K):
            grad_f1 += (grad_xB[:, k] + grad_yB[:, k]) / max((xB[k] + yB[k]) * np.log(2), epsilon) + (grad_yE[:, k]) / max((yE[k] + self.sigma_sq) * np.log(2), epsilon)
            grad_f2 += (grad_yB[:, k]) / max((yB[k]) * np.log(2), epsilon) + (grad_xE[:, k] + grad_yE[:, k]) / max((xE[k] + yE[k] + self.sigma_sq) * np.log(2), epsilon)

        grad_gamma = grad_f1 - grad_f2

        return grad_gamma

    def params(self, C, gamma, p):
        """
        Compute intermediate parameters for gradient calculations.

        Parameters:
        - C: Linear MMSE receive filters.
        - gamma: Current reflection coefficients.
        - p: Power allocation vector for UEs.

        Returns:
        - xB, yB, xE, yE: Intermediate parameters.
        """
        K = p.shape[0]
        N = gamma.shape[0]
        NR_B = self.G_B.shape[0]
        NR_E = self.G_E.shape[1]
<<<<<<< HEAD
        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_e_sq * np.eye(N)
=======
        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_g_sq * np.eye(N)
>>>>>>> origin/main

        xB = np.zeros_like(p)
        for k in range(K):
            ck_B = C[:, k].reshape(NR_B, 1)
            hk = self.H[:, k]
            Hk = np.diag(hk)
            Ak = self.G_B @ Hk
            xB[k] = p[k] * np.sum(np.abs(ck_B.conj().T @ Ak @ gamma) ** 2)

        yB = np.zeros_like(p)
        for k in range(K):
            ck_B = C[:, k].reshape(NR_B, 1)
            pakm_sum_B = 0
            for m in range(K):
                if m != k:
                    hm = self.H[:, m]
                    Hm = np.diag(hm)
                    Am = self.G_B @ Hm
                    akm_B = np.sum(np.abs(ck_B.conj().T @ Am @ gamma) ** 2)
                    pakm_sum_B += p[m] * akm_B
            uk = self.G_B.conj().T @ ck_B
            Uk_tilt = np.diagflat(np.abs(uk) ** 2)
            noise_k = self.sigma_sq * np.linalg.norm(ck_B) ** 2 + self.sigma_RIS_sq * np.linalg.norm(sqrtm(Uk_tilt) @ gamma)**2
            yB[k] = noise_k + pakm_sum_B

        xE = np.zeros_like(p)
        for k in range(K):
            hk = self.H[:, k]
            Hk = np.diag(hk)
            xE[k] = p[k] * np.linalg.norm(sqrtm(RE) @ Hk @ gamma)**2

        yE = np.zeros_like(p)
        for k in range(K):
            interf_m = 0
            for m in range(K):
                if m != k:
                    hm = self.H[:, m]
                    Hm = np.diag(hm)
                    interf_m += p[m] * np.linalg.norm(sqrtm(RE) @ Hm @ gamma) ** 2
            yE[k] = interf_m + self.sigma_RIS_sq * np.linalg.norm(sqrtm(RE) @ gamma)**2

        return xB, yB, xE, yE

    def grad_params(self, C, gamma, p):
        """
        Compute gradients of the intermediate parameters for gradient calculations.

        Parameters:
        - C: Linear MMSE receive filters.
        - gamma: Current reflection coefficients.
        - p: Power allocation vector for UEs.

        Returns:
        - grad_xB, grad_yB, grad_xE, grad_yE: Gradients of the intermediate parameters.
        """
        K = p.shape[0]
        N = gamma.shape[0]
        NR_B = self.G_B.shape[0]
        NR_E = self.G_E.shape[1]
<<<<<<< HEAD
        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_e_sq * np.eye(N)
=======
        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_g_sq * np.eye(N)
>>>>>>> origin/main

        grad_xB = np.zeros((N, K), dtype=np.complex128)
        for k in range(K):
            ck_B = C[:, k].reshape(NR_B, 1)
            hk = self.H[:, k]
            Hk = np.diag(hk)
            Ak = self.G_B @ Hk
            grad_xB[:, k] = 2 * p[k] * ((Ak.conj().T @ (ck_B @ ck_B.conj().T) @ Ak) @ gamma).reshape((N,))

        grad_yB = np.zeros((N, K), dtype=np.complex128)
        for k in range(K):
            ck_B = C[:, k].reshape(NR_B, 1)
            pakm_sum_B = 0
            for m in range(K):
                if m != k:
                    hm = self.H[:, m]
                    Hm = np.diag(hm)
                    Am = self.G_B @ Hm
                    akm_B = (Am.conj().T @ (ck_B @ ck_B.conj().T) @ Am) @ gamma
                    pakm_sum_B += 2 * p[m] * akm_B
            uk = self.G_B.conj().T @ ck_B
            Uk_tilt = np.diagflat(np.abs(uk) ** 2)
            noise_k = 2 * self.sigma_RIS_sq * (Uk_tilt @ gamma)
            grad_yB[:, k] = noise_k.reshape((N,)) + pakm_sum_B.reshape((N,))

        grad_xE = np.zeros((N, K), dtype=np.complex128)
        for k in range(K):
            hk = self.H[:, k]
            Hk = np.diag(hk)
            grad_xE[:, k] = 2 * p[k] * ((Hk.conj().T @ RE @ Hk) @ gamma).reshape((N,))

        grad_yE = np.zeros((N, K), dtype=np.complex128)
        for k in range(K):
            interf_m = 0
            for m in range(K):
                if m != k:
                    hm = self.H[:, m]
                    Hm = np.diag(hm)
                    interf_m += 2 * p[m] * (Hm.conj().T @ RE @ Hm) @ gamma
            grad_yE[:, k] = interf_m.reshape((N,)) + 2 * self.sigma_RIS_sq * (RE @ gamma).reshape((N,))

        return grad_xB, grad_yB, grad_xE, grad_yE

# Example usage:
if __name__ == "__main__":
    # Example parameters (you need to replace these with actual values)
    H = np.random.randn(4, 4)
    G_B = np.random.randn(4, 4)
    G_E = np.random.randn(4, 4)
    sigma_sq = 1e-3
    sigma_RIS_sq = 1e-3
<<<<<<< HEAD
    sigma_e_sq = 1e-3
    mu = 1
    Pc = 1
    scsi_bool = 1
    gamma_utils = GammaUtils(H, G_B, G_E, sigma_sq, sigma_RIS_sq, sigma_e_sq, mu, Pc, scsi_bool)
=======
    sigma_g_sq = 1e-3
    mu = 1
    Pc = 1
    scsi_bool = 1
    gamma_utils = GammaUtils(H, G_B, G_E, sigma_sq, sigma_RIS_sq, sigma_g_sq, mu, Pc, scsi_bool)
>>>>>>> origin/main
    p = np.ones(4)

    C = gamma_utils.LMMSE_receiver_active_Bob(np.ones(4), p)
    gamma_bar = np.ones(4)

    A_bar, B_bar, D_bar, E_bar, F_bar = gamma_utils.parameters_active_Bob(C, gamma_bar, p)
    print("A_bar:", A_bar)
    print("B_bar:", B_bar)
    print("D_bar:", D_bar)
    print("E_bar:", E_bar)
    print("F_bar:", F_bar)

    sr_concave = gamma_utils.SR_active_concave_gamma_Bob(np.ones(4), gamma_bar, p, True)
    print("SR Concave:", sr_concave)
