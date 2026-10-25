import numpy as np
from collections import namedtuple
import torch
from .jackknife import LstsqJackknifeFast


class LD_Score_Regression(object):

    def __init__(self, y, w, N, M, n_blocks=200, intercept=1, twostep=30, step1_ii=None, calculate_se=True):
        """
        y: (chunk_size, n_snp)
        w: (n_snp, )
        """
        x_tot = w.reshape(1, -1)
        w, N = map(lambda a: a.reshape((1, -1)), (w, N))

        self.calculate_se = calculate_se
        self.intercept = intercept
        self.n_blocks = n_blocks
        tot_agg = self.aggregate(y, w, N, M, intercept) # Initinalize h2
        initial_w = self._update_weights(
            x_tot, w, N, M, tot_agg, intercept)
        Nbar = torch.mean(N)  # keep condition number low
        x = N * x_tot / Nbar

        x1, yp1, w1, initial_w1, N1 = map(
            lambda a: a[:,step1_ii], (x, y, w, initial_w, N))
        update_func1 = lambda a: self._update_func(
            a, x1, w1, N1, M, Nbar, ii=step1_ii)
        step1_jknife = self.irwls(
            x1, yp1, update_func1, w=initial_w1, n_blocks=n_blocks)

        if calculate_se:
            step1_int = step1_jknife.est[:, 1]
            yp = y - step1_int.reshape(-1, 1)

            # step1_int = 1
            # yp = y - 1
            s = self.update_separators(step1_jknife.separators, step1_ii)
            update_func2 = lambda a: self._update_func(
                a, x_tot, w, N, M, Nbar, step1_int)
            step2_jknife = self.irwls(
                x, yp, update_func2, w=initial_w, separators=s, n_blocks=n_blocks)
            # print(torch.mean(step1_jknife.est[:,0]*M/Nbar), torch.mean(step2_jknife.est[:,0]*M/Nbar))
            
            c = torch.sum(initial_w * x, dim=1) / torch.sum(initial_w * x.square(), dim=1) # c: (chunk_size,)
            jknife = self._combine_twostep_jknives(
                step1_jknife, step2_jknife, M, c, Nbar)

            self.intercept = jknife.est[:, 1]
            self.slope = jknife.est[:, 0]
            self.cat, self.cat_cov, self.cat_se =\
                self._cat(jknife, M, Nbar, jknife.est[:, 0] / Nbar, jknife.jknife_cov[:, 0, 0] / Nbar ** 2)
            # print(self.cat, self.cat_se)

            # self.tot, self.tot_cov, self.tot_se = self._tot(self.cat, self.cat_cov)
            # self.prop, self.prop_cov, self.prop_se =\
            #     self._prop(jknife, M, Nbar, self.cat, self.tot)

            # self.enrichment, self.M_prop = self._enrichment(
            #     M, M_tot, self.cat, self.tot)
            # if not self.constrain_intercept:
            #     self.intercept, self.intercept_se = self._intercept(jknife)

            # self.jknife = jknife
            self.tot_delete_values = self._delete_vals_tot(jknife, Nbar, M)
            # self.part_delete_values = self._delete_vals_part(jknife, Nbar, M)
            # if not self.constrain_intercept:
            #     self.intercept_delete_values = jknife.delete_values[
            #         :, self.n_annot]

            # self.M = M
        
        else:
            step1_int = step1_jknife[:,1]
            yp = y - step1_int.reshape(-1, 1)
            
            update_func2 = lambda a: self._update_func(
                a, x_tot, w, N, M, Nbar, step1_int)
            step2_jknife = self.irwls(
                x, yp, update_func2, w=initial_w, separators=None, n_blocks=n_blocks)
        
            self.intercept = step1_jknife[:, 1]
            self.slope = step2_jknife[:, 0]
            self.cat = step2_jknife[:, 0] * M / Nbar

    @classmethod
    def aggregate(cls, y, w, N, M, intercept=1):

        num = M * (torch.mean(y, axis=1) - intercept)
        denom = torch.mean(w*N)
        return num / denom


    def irwls(self, x, y, update_func, w, n_iter=2, n_blocks=200, interpret=None, separators=None):
        """
        迭代重加权最小二乘
        
        Parameters:
        -----------
        x : (chunk_size, n) 设计矩阵 [1, ld] * chunk_size
        y : (chunk_size, n) 卡方统计量
        w : (chunk_size, n) 初始权重
        n_iter : int 迭代次数
        interpret: None for step1, and tau from step1 for step2
        
        Returns:
        --------
        tau : 截距
        sigma2 : 斜率
        """
        w.sqrt_()
        chunk_size, m = y.shape
        if separators is None:
            # only add interpret for step 1
            x = torch.vstack([x, torch.ones(m, device=x.device)]).unsqueeze(0).transpose(1,2)
        coef_dim = x.shape[0]

        for i in range(n_iter):
            if separators is None:
                xw = (x * (w / torch.sum(w, axis=1, keepdim=True)).unsqueeze(-1))
            else:
                xw = (x * (w / torch.sum(w, axis=1, keepdim=True))).reshape(chunk_size, m, coef_dim)
            yw = (y * (w / torch.sum(w, axis=1, keepdim=True))).reshape(chunk_size, m, 1)
            coef = torch.linalg.lstsq(xw, yw).solution.squeeze(-1)
            del xw, yw
            
            w = update_func(coef).sqrt_()

        # return coef
        if separators is None:
            x = (x * (w / torch.sum(w, axis=1, keepdim=True)).unsqueeze(-1))
        else:
            x = (x * (w / torch.sum(w, axis=1, keepdim=True))).reshape(chunk_size, m, coef_dim)
        y = (y * (w / torch.sum(w, axis=1, keepdim=True))).reshape(chunk_size, m, 1)
        
        if self.calculate_se:
            jknife = LstsqJackknifeFast(
                    x, y, n_blocks, separators=separators)
            return jknife
        else:
            return coef

    def update_separators(self, s, ii):
        '''s are separators with ii masked. Returns unmasked separators.'''
        maplist = torch.arange(len(ii), device=ii.device)[ii.squeeze()]
        mask_to_unmask = lambda i: maplist[i]
        t = torch.tensor([mask_to_unmask(i) for i in s[1:-1]])
        t = torch.cat((torch.tensor([0]), t, torch.tensor([len(ii)])))
        return t

    def _combine_twostep_jknives(self, step1_jknife, step2_jknife, M_tot, c, Nbar=1):
        '''Combine free intercept and constrained intercept jackknives for --two-step.'''
        chunk_size, n_blocks, n_annot = step1_jknife.delete_values.shape
        n_annot = n_annot - 1
        if n_annot > 2:
            raise ValueError(
                'twostep not yet implemented for partitioned LD Score.')

        step1_int = step1_jknife.est[:, 1].reshape(-1,1)  # (chunk_size, 1)
        
        # est: (chunk_size, n_annot + 1)
        est = torch.cat((step2_jknife.est, step1_int), dim=1)
        
        # delete_values: (chunk_size, n_blocks, n_annot + 1)
        delete_values = torch.zeros((chunk_size, n_blocks, n_annot + 1), 
                                    device=step1_jknife.delete_values.device,
                                    dtype=step1_jknife.delete_values.dtype)
        delete_values[:, :, n_annot] = step1_jknife.delete_values[:, :, n_annot]
        
        # c shape: (chunk_size, 1) for broadcasting
        c_expanded = c.unsqueeze(-1)  # (chunk_size, 1)
        step1_int_expanded = step1_int.unsqueeze(1)  # (chunk_size, 1, 1)
        step1_delete = step1_jknife.delete_values[:, :, n_annot].unsqueeze(-1)  # (chunk_size, n_blocks, 1)
        
        delete_values[:, :, 0:n_annot] = step2_jknife.delete_values - \
            c_expanded.unsqueeze(1) * (step1_delete - step1_int_expanded)
        
        pseudovalues = LstsqJackknifeFast.delete_values_to_pseudovalues(
            delete_values, est)
        jknife_est, jknife_var, jknife_se, jknife_cov = LstsqJackknifeFast.jknife(
            pseudovalues)
        
        jknife = namedtuple('jknife',
                            ['est', 'jknife_se', 'jknife_est', 'jknife_var', 'jknife_cov', 'delete_values'])
        return jknife(est, jknife_se, jknife_est, jknife_var, jknife_cov, delete_values)

    def _cat(self, jknife, M, Nbar, coef, coef_cov):
        '''Convert coefficients to per-category h2 or gencov.'''
        cat = M * coef  # (chunk_size, n_categories)
        cat_cov = (M ** 2) * coef_cov  # (chunk_size, n_categories)
        cat_se = torch.sqrt(cat_cov)  # (chunk_size, n_categories)
        return cat, cat_cov, cat_se

    def _delete_vals_tot(self, jknife, Nbar, M):
        '''Get delete values for total h2 or gencov.'''
        tot_delete_vals = jknife.delete_values[:,:,0]  # shape (chunk_size, n_blocks)
        tot_delete_vals = tot_delete_vals * M / Nbar
        return tot_delete_vals



class Hsq(LD_Score_Regression):

    __null_intercept__ = 1

    def __init__(self, y, w, N, M, n_blocks=200, intercept=None, twostep=None, calculate_se=True):
        step1_ii = None
        if twostep is not None:
            step1_ii = (torch.max(y, axis=0)[0]<=twostep)

        LD_Score_Regression.__init__(self, y, w, N, M, n_blocks, intercept=intercept,
                                     step1_ii=step1_ii, calculate_se=calculate_se)

    def _update_weights(self, ld, w_ld, N, M, hsq, intercept):
        '''Weight function with the same signature for Hsq and Gencov.'''
        w = self.weights(ld, w_ld, N, M, hsq, intercept)
        return w

    @classmethod
    def weights(self, ld, w_ld, N, M, hsq, intercept=None):
        '''
        Regression weights (PyTorch version).
        
        Parameters:
        -----------
        ld : torch.Tensor, shape (1, n_snp)
            LD Scores
        w_ld : torch.Tensor, shape (1, n_snp)
            LD Scores for regression
        N : torch.Tensor, shape (1, n_snp)
            Sample size per SNP
        M : float
            Number of SNPs
        hsq : torch.Tensor, shape (chunk_size,)
            Heritability estimates for each voxel
        intercept : torch.Tensor, shape (chunk_size,), optional
            Intercept for each voxel, default 1
        
        Returns:
        --------
        w : torch.Tensor, shape (chunk_size, n_snp)
            Regression weights
        '''
        if intercept is None:
            intercept = torch.ones_like(hsq)
        elif not isinstance(intercept, torch.Tensor):
            intercept = torch.full_like(hsq, float(intercept))
        intercept = intercept.to(torch.float32)
        
        hsq = torch.clamp(hsq, 0.0, 1.0)
        ld = torch.maximum(ld, torch.ones_like(ld))
        w_ld = torch.maximum(w_ld, torch.ones_like(w_ld))
        
        # c = hsq * N / M
        # intercept + c * ld
        # intercept: (chunk_size,), c: (n_snp, chunk_size), ld: (n_snp, 1)
        denom = intercept.unsqueeze(1) + (hsq.unsqueeze(1) * N / M) * ld
        # het_w = 1 / (2 * denom^2)
        het_w = denom.pow_(2).mul_(2.0).reciprocal_()
        # oc_w = 1 / w_ld
        oc_w = 1.0 / w_ld
        # w = het_w * oc_w
        w = het_w * oc_w

        del denom, het_w, oc_w
        return w

    def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar, intercept=None, ii=None):
        '''
        Update function for IRWLS

        x is the output of np.linalg.lstsq.
        x is the regression coefficients
        x.shape is (# chunk_size, 2)
        x[:,0] is the slope x[:,1] is the intercept.

        intercept is None --> free intercept
        intercept is not None --> constrained intercept
        '''
        hsq = M * x[:,0] / Nbar
        if intercept is None:
            intercept = x[:,1]  # divide by zero error if intercept < 0
        else:
            if ref_ld_tot.shape[0] > 1:
                raise ValueError(
                    'Design matrix has intercept column for constrained intercept regression!')

        ld = ref_ld_tot.reshape(w_ld.shape)  # remove intercept
        w = self.weights(ld, w_ld, N, M, hsq, intercept)
        return w



class Gencov(LD_Score_Regression):
    __null_intercept__ = 0

    def __init__(self, z1, z2, w, N1, N2, M, hsq1, hsq2, intercept_hsq1, intercept_hsq2,
                 n_blocks=200, intercept_gencov=None, twostep=None, calculate_se=True):
        self.intercept_hsq1 = intercept_hsq1
        self.intercept_hsq2 = intercept_hsq2
        self.hsq1 = hsq1
        self.hsq2 = hsq2
        self.N1 = N1
        self.N2 = N2
        y = z1 * z2.unsqueeze(0)
        step1_ii = None
        if twostep is not None:
            z1_sq = z1 ** 2  # (n_voxels, n_snps)
            z2_sq = (z2 ** 2)  # (n_snps,)

            max_z1_sq_per_snp = torch.max(z1_sq, dim=0)[0]  # (n_snps,)
            step1_ii = (max_z1_sq_per_snp < twostep) & (z2_sq < twostep)  # (n_snps,)

        LD_Score_Regression.__init__(self, y, w, torch.sqrt(N1 * N2), M, n_blocks,
                                     intercept=intercept_gencov, step1_ii=step1_ii, calculate_se=calculate_se)
        # self.p, self.z = p_z_norm(self.tot, self.tot_se)
        # self.mean_z1z2 = np.mean(np.multiply(z1, z2))

    def _update_func(self, x, ref_ld_tot, w_ld, N, M, Nbar, intercept=None, ii=None):
        '''
        Update function for IRWLS

        x is the output of np.linalg.lstsq.
        x is the regression coefficients
        x.shape is (# chunk_size, 2)
        x[:,0] is the slope x[:,1] is the intercept.

        intercept is None --> free intercept
        intercept is not None --> constrained intercept

        '''
        rho_g = M * x[:,0] / Nbar
        if intercept is None:  # if the regression includes an intercept
            intercept = x[:, 1]

        # remove intercept if we have one
        ld = ref_ld_tot.reshape(w_ld.shape)  # remove intercept
        if ii is not None:
            N1 = self.N1[ii]
            N2 = self.N2[ii]
        else:
            N1 = self.N1
            N2 = self.N2

        return self.weights(ld, w_ld, N1, N2, np.sum(M), self.hsq1, self.hsq2, rho_g,
                         intercept, self.intercept_hsq1, self.intercept_hsq2)

    def _update_weights(self, ld, w_ld, sqrt_n1n2, M, rho_g, intercept, ii=None):
        '''Weight function with the same signature for Hsq and Gencov.'''
        if ii is not None:
            N1 = self.N1[ii]
            N2 = self.N2[ii]
        else:
            N1 = self.N1
            N2 = self.N2

        w = self.weights(ld, w_ld, N1, N2, M, self.hsq1, self.hsq2, rho_g,
                         intercept, self.intercept_hsq1, self.intercept_hsq2)
        return w

    @classmethod
    def weights(cls, ld, w_ld, N1, N2, M, h1, h2, rho_g, intercept_gencov=None,
                intercept_hsq1=None, intercept_hsq2=None):
        """
        回归权重计算
        
        Parameters:
        -----------
        ld : torch.Tensor, shape (n_snps,)
            LD 分数
        w_ld : torch.Tensor, shape (n_snps,)
            回归权重 LD 分数
        N1 : torch.Tensor, shape (n_snps,)
            体素级 GWAS 样本量
        N2 : torch.Tensor, shape (n_snps,)
            pheno GWAS 样本量
        M : float
            SNP 数量
        h1 : torch.Tensor, shape (n_voxels,)
            体素级遗传力
        h2 : float
            pheno 遗传力
        rho_g : torch.Tensor, shape (n_voxels,)
            遗传协方差
        intercept_gencov : torch.Tensor, shape (n_voxels,), optional
            遗传协方差截距
        intercept_hsq1 : torch.Tensor, shape (n_voxels,), optional
            体素级遗传力截距
        intercept_hsq2 : float, optional
            pheno 遗传力截距
        
        Returns:
        --------
        w : torch.Tensor, shape (n_snps, n_voxels)
            回归权重
        """
        # 设置默认值
        if intercept_gencov is None:
            intercept_gencov = torch.zeros_like(h1)
        elif not isinstance(intercept_gencov, torch.Tensor):
            intercept_gencov = torch.full_like(h1, float(intercept_gencov))
        
        if intercept_hsq1 is None:
            intercept_hsq1 = torch.ones_like(h1)
        elif not isinstance(intercept_hsq1, torch.Tensor):
            intercept_hsq1 = torch.full_like(h1, float(intercept_hsq1))
        else:
            intercept_hsq1 = intercept_hsq1.reshape(-1,1)
        
        if intercept_hsq2 is None:
            intercept_hsq2 = torch.ones_like(h1)
        elif not isinstance(intercept_hsq2, torch.Tensor):
            intercept_hsq2 = torch.full_like(h1, float(intercept_hsq2))
        
        M = float(M)
        h1 = torch.clamp(h1, 0.0, 1.0)  # (n_voxels, 1)
        h2 = torch.clamp(h2, 0.0, 1.0)
        rho_g = torch.clamp(rho_g, -1.0, 1.0).unsqueeze(1)  # (n_voxels, 1)
        
        ld = torch.maximum(ld, torch.ones_like(ld))  # (n_snps,)
        w_ld = torch.maximum(w_ld, torch.ones_like(w_ld))  # (n_snps,)
        
        a = N1.unsqueeze(0) * h1 * ld / M + intercept_hsq1 # (n_voxels, n_snps)
        # b = N2 * h2 * ld / M + intercept_hsq2
        b = N2.unsqueeze(0) * h2 * ld / M + intercept_hsq2 # (n_voxels, n_snps)
        # c = sqrt(N1 * N2) * rho_g * ld / M + intercept_gencov
        sqrt_n1n2 = torch.sqrt(N1 * N2).unsqueeze(0)  # (1, n_snps)
        c = sqrt_n1n2 * rho_g * ld / M + rho_g
        # het_w = 1 / (a * b + c^2)
        het_w = 1.0 / (a * b + c ** 2)  # (n_snps, n_voxels)
        # oc_w = 1 / w_ld
        oc_w = 1.0 / w_ld  # (1, n_snps)
        w = het_w * oc_w  # (n_snps, n_voxels)
        
        return w

