import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm
from utils import Utils

class PowerUtils:

    def __init__(self, H, G_B, G_E, sigma_sq, sigma_RIS_sq, sigma_e_sq, mu, Pc, scsi_bool=1, utils_cls=Utils):

        """
        Initialize the PowerUtils class with given parameters.

        Parameters:
        - H: Channel matrix from UEs to RIS.
        - G_B: Channel matrix from RIS to Bob.
        - g_E: Channel matrix from RIS to Eve.
        - sigma_sq: Noise power variance at the receiver.
        - sigma_RIS_sq: Noise power variance at RIS.
        - sigma_g_sq: Noise variance for Eve's channel estimation error.
        - mu: Amplifier inefficiency factor.
        - Pc: Static power consumption.
        - scsi_bool: Boolean indicating whether to consider channel state information (CSI).
        - utils_cls: Utility class for common functions.
        """
        self.H = H
        self.G_B = G_B
        self.G_E = G_E
        self.sigma_sq = sigma_sq
        self.sigma_RIS_sq = sigma_RIS_sq
        self.sigma_e_sq = sigma_e_sq
        self.mu = mu
        self.Pc = Pc
        self.scsi_bool = scsi_bool
        self.utils_cls = utils_cls

    def compute_Pc_eq_p(self, gamma):
        """
        Compute the equivalent power consumption considering gamma.

        Parameters:
        - gamma: Current reflection coefficients.

        Returns:
        - Pc_eq: Equivalent power consumption.
        """
        N = gamma.shape[0]
        Pc_eq = self.sigma_RIS_sq * (np.linalg.norm(gamma)**2 - N) + self.Pc
        return Pc_eq

    def compute_mu_eq_p(self, gamma):
        """
        Compute the equivalent amplifier inefficiency factor.

        Parameters:
        - gamma: Current reflection coefficients.

        Returns:
        - mu_eq: Equivalent amplifier inefficiency factor.
        """
        K = self.H.shape[1]
        mu_eq = np.zeros(K)

        for k in range(K):
            Hk = np.diag(self.H[:, k])
            norm_hk_sq = np.linalg.norm(self.H[:, k]) ** 2
            mu_eq[k] = self.mu - norm_hk_sq + np.linalg.norm(Hk @ gamma) ** 2

        return mu_eq

    def compute_g1(self, C, gamma, p):
        """
        Compute the first optimization function g1.

        Parameters:
        - C: Linear MMSE receive filters.
        - gamma: Current reflection coefficients.
        - p: Power allocation vector for UEs.

        Returns:
        - g1: Value of the optimization function g1.
        """
        K = p.shape[0]
        NR_B = self.G_B.shape[0]
        N = self.H.shape[0]
        epsilon = np.finfo(float).eps

        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_e_sq * np.eye(N)

        d_E = self.sigma_RIS_sq * np.linalg.norm(sqrtm(RE) @ gamma)**2 + self.sigma_sq
        g1 = 0
        Gamma = np.diagflat(gamma)
        W_B = self.sigma_sq * np.eye(NR_B) + self.sigma_RIS_sq * self.G_B @ (Gamma @ Gamma.conj().T) @ self.G_B.conj().T

        for k in range(K):
            ck_B = C[:, k].reshape(NR_B, 1)
            dk_B = np.real(np.sum(ck_B.conj().T @ W_B @ ck_B))
            pakk_sum_B = 0
            pam_sum_E = 0

            for m in range(K):
                hm = self.H[:, m]
                Hm = np.diag(hm)
                Am_B = self.G_B @ Hm
                akm_B = np.sum(np.abs(ck_B.conj().T @ Am_B @ gamma) ** 2)

                pakk_sum_B += p[m] * akm_B / max(dk_B, epsilon)
                if m != k:
                    am_E = np.linalg.norm(sqrtm(RE) @ Hm @ gamma) ** 2
                    pam_sum_E += p[m] * am_E / max(d_E, epsilon)

            g1 += cp.inv_pos(cp.log(2)) * cp.log(1 + pakk_sum_B) + cp.inv_pos(cp.log(2)) * cp.log(1 + pam_sum_E)

        return g1

    def compute_g2(self, C, gamma, p):
        """
        Compute the second optimization function g2.

        Parameters:
        - C: Linear MMSE receive filters.
        - gamma: Current reflection coefficients.
        - p: Power allocation vector for UEs.

        Returns:
        - g2: Value of the optimization function g2.
        """
        K = p.shape[0]
        NR_B = self.G_B.shape[0]
        N = self.H.shape[0]
        epsilon = np.finfo(float).eps

        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_e_sq * np.eye(N)

        d_E = self.sigma_RIS_sq * np.linalg.norm(sqrtm(RE) @ gamma)**2 + self.sigma_sq
        g2 = 0
        Gamma = np.diagflat(gamma)
        W_B = self.sigma_sq * np.eye(NR_B) + self.sigma_RIS_sq * self.G_B @ (Gamma @ Gamma.conj().T) @ self.G_B.conj().T

        for k in range(K):
            ck_B = C[:, k].reshape(NR_B, 1)
            dk_B = np.real(np.sum(ck_B.conj().T @ W_B @ ck_B))
            pakm_sum_B = 0
            pam_sum_E = 0

            for m in range(K):
                hm = self.H[:, m]
                Hm = np.diag(hm)
                am_E = np.linalg.norm(sqrtm(RE) @ Hm @ gamma) ** 2
                pam_sum_E += p[m] * am_E / max(d_E, epsilon)
                if m != k:
                    Am_B = self.G_B @ Hm
                    akm_B = np.sum(np.abs(ck_B.conj().T @ Am_B @ gamma) ** 2)
                    pakm_sum_B += p[m] * akm_B / max(dk_B, epsilon)

            g2 += cp.inv_pos(cp.log(2)) * cp.log(1 + pakm_sum_B) + cp.inv_pos(cp.log(2)) * cp.log(1 + pam_sum_E)

        return g2

    def compute_gradg1(self, C, gamma, p):
        """
        Compute the gradient of g1 with respect to gamma.

        Parameters:
        - C: Linear MMSE receive filters.
        - gamma: Current reflection coefficients.
        - p: Power allocation vector for UEs.

        Returns:
        - grad_g1: Gradient of g1 with respect to gamma.
        """
        K = p.shape[0]
        NR_B = self.G_B.shape[0]
        N = self.H.shape[0]
        epsilon = np.finfo(float).eps

        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_e_sq * np.eye(N)

        d_E = self.sigma_RIS_sq * np.linalg.norm(sqrtm(RE) @ gamma)**2 + self.sigma_sq

        grad_g1 = np.zeros_like(p)
        grad_g1_B = np.zeros_like(p)
        grad_g1_E = np.zeros_like(p)

        Gamma = np.diagflat(gamma)
        W_B = self.sigma_sq * np.eye(NR_B) + self.sigma_RIS_sq * self.G_B @ (Gamma @ Gamma.conj().T) @ self.G_B.conj().T

        for i in range(K):
            hi = self.H[:, i]
            Hi = np.diag(hi)
            Ai_B = self.G_B @ Hi
            ai_E = np.linalg.norm(sqrtm(RE) @ Hi @ gamma) ** 2

            for k in range(K):
                ck_B = C[:, k].reshape(NR_B, 1)
                dk_B = np.real(ck_B.conj().T @ W_B @ ck_B)
                aki_B = np.sum(np.abs(ck_B.conj().T @ Ai_B @ gamma) ** 2)

                pakm_sum_B = 0

                for m in range(K):
                    hm = self.H[:, m]
                    Hm = np.diag(hm)
                    Am_B = self.G_B @ Hm
                    akm_B = np.sum(np.abs(ck_B.conj().T @ Am_B @ gamma)**2)
                    pakm_sum_B += p[m] * akm_B

                grad_g1_B[i] += aki_B / max((dk_B + pakm_sum_B) * np.log(2), epsilon)

                if k != i:
                    pam_sum_E = 0
                    for m in range(K):
                        if m != k:
                            hm = self.H[:, m]
                            Hm = np.diag(hm)
                            am_E = np.linalg.norm(sqrtm(RE) @ Hm @ gamma) ** 2
                            pam_sum_E += p[m] * am_E

                    grad_g1_E[i] += ai_E / max((d_E + pam_sum_E) * np.log(2), epsilon)

            grad_g1[i] = grad_g1_B[i] + grad_g1_E[i]

        return grad_g1

    def compute_gradg2(self, C, gamma, p):
        """
        Compute the gradient of g2 with respect to gamma.

        Parameters:
        - C: Linear MMSE receive filters.
        - gamma: Current reflection coefficients.
        - p: Power allocation vector for UEs.

        Returns:
        - grad_g2: Gradient of g2 with respect to gamma.
        """
        K = p.shape[0]
        NR_B = self.G_B.shape[0]
        N = self.H.shape[0]
        epsilon = np.finfo(float).eps

        RE = self.G_E @ self.G_E.conj().T + self.scsi_bool * self.sigma_e_sq * np.eye(N)

        d_E = self.sigma_RIS_sq * np.linalg.norm(sqrtm(RE) @ gamma)**2 + self.sigma_sq

        grad_g2 = np.zeros_like(p)
        grad_g2_B = np.zeros_like(p)
        grad_g2_E = np.zeros_like(p)

        Gamma = np.diagflat(gamma)
        W_B = self.sigma_sq * np.eye(NR_B) + self.sigma_RIS_sq * self.G_B @ (Gamma @ Gamma.conj().T) @ self.G_B.conj().T
        
        pam_sum_E = 0
        for m in range(K):
            hm = self.H[:, m]
            Hm = np.diag(hm)
            am_E = np.linalg.norm(sqrtm(RE) @ Hm @ gamma) ** 2
            pam_sum_E += p[m] * am_E

        for i in range(K):
            hi = self.H[:, i]
            Hi = np.diag(hi)
            Ai_B = self.G_B @ Hi
            ai_E = np.linalg.norm(sqrtm(RE) @ Hi @ gamma) ** 2

            for k in range(K):
                if k != i:
                    ck_B = C[:, k].reshape(NR_B, 1)
                    dk_B = np.real(ck_B.conj().T @ W_B @ ck_B)
                    aki_B = np.sum(np.abs(ck_B.conj().T @ Ai_B @ gamma) ** 2)

                    pakm_sum_B = 0

                    for m in range(K):
                        if m != k:
                            hm = self.H[:, m]
                            Hm = np.diag(hm)
                            Am_B = self.G_B @ Hm
                            akm_B = np.sum(np.abs(ck_B.conj().T @ Am_B @ gamma)**2)
                            pakm_sum_B += p[m] * akm_B

                    grad_g2_B[i] += aki_B / max((dk_B + pakm_sum_B) * np.log(2), epsilon)

                grad_g2_E[i] += ai_E / max((d_E + pam_sum_E) * np.log(2), epsilon)

            grad_g2[i] = grad_g2_B[i] + grad_g2_E[i]

        return grad_g2

    def compute_grad_p(self, C, gamma, p):
        """
        Compute the gradient of the objective function with respect to p.

        Parameters:
        - C: Linear MMSE receive filters.
        - gamma: Current reflection coefficients.
        - p: Power allocation vector for UEs.

        Returns:
        - grad_p: Gradient of the objective function with respect to p.
        """
        grad_g1 = self.compute_gradg1(C, gamma, p)
        grad_g2 = self.compute_gradg2(C, gamma, p)
        grad_p = grad_g1 - grad_g2
        return grad_p

    def SSR_active_concave_p(self, gamma, p, p_bar, cvx_bool):
        """
        Compute the concave approximation of the secrecy rate with respect to p.

        Parameters:
        - gamma: Current reflection coefficients.
        - p: Power allocation vector for UEs.
        - p_bar: Previous power allocation vector for UEs.
        - cvx_bool: Boolean indicating whether to use convex approximation.

        Returns:
        - srp_concave: Concave approximation of the secrecy rate.
        """
        C_B = self.utils_cls.LMMSE_receiver_active_Bob(self.G_B, self.H, gamma, p_bar, self.sigma_sq, self.sigma_RIS_sq)

        g1 = self.compute_g1(C_B, gamma, p)
        g2_bar = self.compute_g2(C_B, gamma, p_bar)
        grad_g2 = self.compute_gradg2(C_B, gamma, p_bar)

        srp_concave = g1 - g2_bar * cvx_bool - cp.sum(cp.matmul(grad_g2, (p - p_bar * cvx_bool).T))
        return srp_concave

# Example usage
if __name__ == "__main__":
    # Example parameters (replace these with actual values)
    H = np.random.randn(4, 4)
    G_B = np.random.randn(4, 4)
    g_E = np.random.randn(4, 4)
    sigma_sq = 1e-3
    sigma_RIS_sq = 1e-3
    sigma_e_sq = 1e-3
    mu = 1
    Pc = 1
    scsi_bool = 1
    power_utils = PowerUtils(H, G_B, g_E, sigma_sq, sigma_RIS_sq, sigma_e_sq, mu, Pc, scsi_bool)

    gamma = np.ones(4)
    p = np.ones(4)
    p_bar = np.ones(4)

    C = power_utils.utils_cls.LMMSE_receiver_active_Bob(G_B, H, gamma, p_bar, sigma_sq, sigma_RIS_sq)
    srp_concave = power_utils.SSR_active_concave_p(gamma, p, p_bar, True)
    print("SRP Concave:", srp_concave)
