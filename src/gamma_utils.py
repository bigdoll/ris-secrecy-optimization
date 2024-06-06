import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm
from utils import Utils

class GammaUtils:
    def __init__(self, H, G_B, G_E, sigma_sq, sigma_RIS_sq, sigma_g_sq, mu, Pc, scsi_bool=1, utils_cls=Utils):
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
        self.sigma_g_sq = sigma_g_sq
        self.mu = mu
        self.Pc = Pc
        self.scsi_bool = scsi_bool
        self.utils_cls = utils_cls
    
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

        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_g_sq * np.eye(N)

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
        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_g_sq * np.eye(N)
        sr_concave = 0

        A_bar, B_bar, D_bar, E_bar, F_bar, L_bar, Q_bar, V_bar = self.parameters_active_Eve(gamma_bar, p)

        for k in range(K):
            xE = p[k] * cp.real(cp.sum(cp.quad_form(gamma, self.H[:, k].conj().T @ RE @ self.H[:, k])))
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

            yE = interf_m + self.sigma_RIS_sq * cp.real(cp.sum(cp.quad_form(gamma, RE)))
            yE_bar = interf_m_bar + self.sigma_RIS_sq * np.linalg.norm(sqrtm(RE) @ gamma_bar)**2
            yE_bar_sqrt = np.sqrt(yE_bar)
            grad_yE_bar = interf_m_grad + self.sigma_RIS_sq * RE @ gamma_bar
            grad_yE_sqrt_bar = grad_yE_bar / np.where(yE_bar_sqrt > epsilon, yE_bar_sqrt, epsilon)

            sr_concave += A_bar[k] * cvx_bool + (B_bar[k] * D_bar[k]) * (yE_bar_sqrt * cvx_bool + cp.real(cp.sum(cp.matmul(grad_yE_sqrt_bar.conj().T, (gamma - gamma_bar * cvx_bool))))) - (B_bar[k] * F_bar[k]) * cvx_bool - (B_bar[k] * E_bar[k]) * yE + (L_bar[k] + V_bar[k]/np.log(2)) * cvx_bool - (Q_bar[k]/np.log(2)) * (xE + yE)

        return sr_concave

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
        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_g_sq * np.eye(N)

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
        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_g_sq * np.eye(N)

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
    sigma_g_sq = 1e-3
    mu = 1
    Pc = 1
    scsi_bool = 1
    gamma_utils = GammaUtils(H, G_B, G_E, sigma_sq, sigma_RIS_sq, sigma_g_sq, mu, Pc, scsi_bool)
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
