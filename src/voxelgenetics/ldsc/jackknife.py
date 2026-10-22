import numpy as np
import torch


class LstsqJackknifeFast(object):
    '''
    Fast block jackknife for linear regression.

    Parameters
    ----------
    x : torch.Tensor with shape (chunk_size, n, p)
        Independent variable.
    y : torch.Tensor with shape (chunk_size, n, 1)
        Dependent variable.
    n_blocks : int
        Number of jackknife blocks
    separators : list of ints, optional
        Block separators.

    Attributes
    ----------
    est : torch.Tensor with shape (chunk_size, p)
        FWLS estimate.
    jknife_est : torch.Tensor with shape (chunk_size, p)
        Jackknifed estimate.
    jknife_var : torch.Tensor with shape (chunk_size, p)
        Variance of jackknifed estimate.
    jknife_se : torch.Tensor with shape (chunk_size, p)
        Standard error of jackknifed estimate, equal to sqrt(jknife_var).
    jknife_cov : torch.Tensor with shape (chunk_size, p, p)
        Covariance matrix of jackknifed estimate.
    delete_values : torch.Tensor with shape (chunk_size, n_blocks, p)
        Jackknife delete values.
    pseudovalues : torch.Tensor with shape (chunk_size, n_blocks, p)
        Pseudovalues.
    '''

    def __init__(self, x, y, n_blocks=None, separators=None):
        self.n_blocks = n_blocks
        N = x.shape[1]
        if separators is None:
            self.separators = self.get_separators(N, self.n_blocks)
        else:
            self.separators = separators

        xty, xtx = self.block_values(x, y, self.separators)
        self.est = self.block_values_to_est(xty, xtx)
        self.delete_values = self.block_values_to_delete_values(xty, xtx)
        self.pseudovalues = self.delete_values_to_pseudovalues(
            self.delete_values, self.est)
        (self.jknife_est, self.jknife_var, self.jknife_se, self.jknife_cov) =\
            self.jknife(self.pseudovalues)

    @classmethod
    def get_separators(cls, N, n_blocks):
        '''Define evenly-spaced block boundaries.'''
        return torch.floor(torch.linspace(0, N, n_blocks + 1)).to(dtype=torch.long)

    @classmethod
    def block_values(cls, x, y, s):
        '''
        Compute block values.

        Parameters
        ----------
        x : torch.Tensor with shape (chunk_size, n, p)
            Independent variable.
        y : torch.Tensor with shape (chunk_size, n, 1)
            Dependent variable.
        s : torch.Tensor of ints
            Block separators.

        Returns
        -------
        xty_block_values : torch.Tensor with shape (chunk_size, n_blocks, p)
            Block values of X^T Y.
        xtx_block_values : torch.Tensor with shape (chunk_size, n_blocks, p, p)
            Block values of X^T X.
        '''
        chunk_size, n, p = x.shape
        n_blocks = len(s) - 1
        device = x.device
        dtype = x.dtype
        
        xtx_block_values = torch.zeros((chunk_size, n_blocks, p, p), device=device, dtype=dtype)
        xty_block_values = torch.zeros((chunk_size, n_blocks, p), device=device, dtype=dtype)
        
        for i in range(n_blocks):
            x_block = x[:, s[i]:s[i+1], :]
            y_block = y[:, s[i]:s[i+1], :]
            
            xty = torch.matmul(x_block.transpose(1, 2), y_block)
            xty_block_values[:, i, :] = xty.squeeze(-1)
            
            xtx = torch.matmul(x_block.transpose(1, 2), x_block)
            xtx_block_values[:, i, :, :] = xtx
        
        return xty_block_values, xtx_block_values

    @classmethod
    def block_values_to_est(cls, xty_block_values, xtx_block_values):
        '''
        Converts block values to the whole-data linear regression estimate.

        Parameters
        ----------
        xty_block_values : torch.Tensor with shape (chunk_size, n_blocks, p)
            Block values of X^T Y.
        xtx_block_values : torch.Tensor with shape (chunk_size, n_blocks, p, p)
            Block values of X^T X

        Returns
        -------
        est : torch.Tensor with shape (chunk_size, p)
            Whole data estimate.
        '''
        xty = xty_block_values.sum(dim=1)
        xtx = xtx_block_values.sum(dim=1)
        
        xty_expanded = xty.unsqueeze(-1)
        
        try:
            est = torch.linalg.solve(xtx, xty_expanded)
            est = est.squeeze(-1)
        except RuntimeError:
            est = torch.linalg.pinv(xtx) @ xty_expanded
            est = est.squeeze(-1)
        
        return est

    @classmethod
    def block_values_to_delete_values(cls, xty_block_values, xtx_block_values):
        '''
        Converts block values to delete values.

        Parameters
        ----------
        xty_block_values : torch.Tensor with shape (chunk_size, n_blocks, p)
            Block values of X^T Y.
        xtx_block_values : torch.Tensor with shape (chunk_size, n_blocks, p, p)
            Block values of X^T X

        Returns
        -------
        delete_values : torch.Tensor with shape (chunk_size, n_blocks, p)
            Delete Values.
        '''
        chunk_size, n_blocks, p = xty_block_values.shape
        device = xty_block_values.device
        dtype = xty_block_values.dtype
        
        xty_tot = xty_block_values.sum(dim=1)
        xtx_tot = xtx_block_values.sum(dim=1)
        
        delete_values = torch.zeros((chunk_size, n_blocks, p), device=device, dtype=dtype)
        
        for j in range(n_blocks):
            delete_xty = xty_tot - xty_block_values[:, j, :]
            delete_xtx = xtx_tot - xtx_block_values[:, j, :, :]
            
            delete_xty_expanded = delete_xty.unsqueeze(-1)
            
            try:
                delete_est = torch.linalg.solve(delete_xtx, delete_xty_expanded)
                delete_values[:, j, :] = delete_est.squeeze(-1)
            except RuntimeError:
                delete_est = torch.linalg.pinv(delete_xtx) @ delete_xty_expanded
                delete_values[:, j, :] = delete_est.squeeze(-1)
        
        return delete_values

    @classmethod
    def delete_values_to_pseudovalues(cls, delete_values, est):
        '''
        Converts whole-data estimate and delete values to pseudovalues.

        Parameters
        ----------
        delete_values : torch.Tensor with shape (chunk_size, n_blocks, p)
            Delete values.
        est : torch.Tensor with shape (chunk_size, p)
            Whole-data estimate.

        Returns
        -------
        pseudovalues : torch.Tensor with shape (chunk_size, n_blocks, p)
            Pseudovalues.
        '''
        chunk_size, n_blocks, p = delete_values.shape
        
        est_expanded = est.unsqueeze(1)
        
        pseudovalues = n_blocks * est_expanded - (n_blocks - 1) * delete_values
        
        return pseudovalues

    @classmethod
    def jknife(cls, pseudovalues):
        '''
        Converts pseudovalues to jackknife estimate and variance.

        Parameters
        ----------
        pseudovalues : torch.Tensor with shape (chunk_size, n_blocks, p)

        Returns
        -------
        jknife_est : torch.Tensor with shape (chunk_size, p)
            Jackknifed estimate.
        jknife_var : torch.Tensor with shape (chunk_size, p)
            Variance of jackknifed estimate.
        jknife_se : torch.Tensor with shape (chunk_size, p)
            Standard error of jackknifed estimate, equal to sqrt(jknife_var).
        jknife_cov : torch.Tensor with shape (chunk_size, p, p)
            Covariance matrix of jackknifed estimate.
        '''
        chunk_size, n_blocks, p = pseudovalues.shape
        jknife_est = pseudovalues.mean(dim=1)  # (chunk_size, p)
        centered = pseudovalues - jknife_est.unsqueeze(1)  # (chunk_size, n_blocks, p)
        jknife_cov = torch.matmul(centered.transpose(1, 2), centered) / (n_blocks - 1)  # (chunk_size, p, p)
        jknife_cov = jknife_cov / n_blocks  # (chunk_size, p, p)
        jknife_var = torch.diagonal(jknife_cov, dim1=1, dim2=2)  # (chunk_size, p)
        jknife_se = torch.sqrt(jknife_var)  # (chunk_size, p)
        
        return jknife_est, jknife_var, jknife_se, jknife_cov



class RatioJackknife(object):
    '''
    Block jackknife ratio estimate (PyTorch version with batch support).

    Parameters
    ----------
    est : torch.Tensor with shape (chunk_size, p) or (p,)
        Whole data ratio estimate
    numer_delete_values : torch.Tensor with shape (chunk_size, n_blocks, p) or (n_blocks, p)
        Delete values for the numerator.
    denom_delete_values : torch.Tensor with shape (chunk_size, n_blocks, p) or (n_blocks, p)
        Delete values for the denominator.

    Methods
    -------
    delete_values_to_pseudovalues(est, denom, numer):
        Converts denominator/numerator delete values and the whole-data estimate to
        pseudovalues.

    Raises
    ------
    FloatingPointError :
        If any entry of denom_delete_values is zero.
    '''

    def __init__(self, est, numer_delete_values, denom_delete_values):
        # 统一为 3D 张量: (chunk_size, n_blocks, p)
        # if numer_delete_values.dim() == 2:
        #     # (n_blocks, p) -> (1, n_blocks, p)
        #     numer_delete_values = numer_delete_values.unsqueeze(0)
        #     denom_delete_values = denom_delete_values.unsqueeze(0)
        #     est = est.unsqueeze(0) if est.dim() == 1 else est
        
        if numer_delete_values.shape != denom_delete_values.shape:
            raise ValueError(
                'numer_delete_values.shape != denom_delete_values.shape.')
        
        if numer_delete_values.dim() != 2:
            raise ValueError('Delete values must be 2D tensors (chunk_size, n_blocks).')

        self.chunk_size = numer_delete_values.shape[0]
        self.n_blocks = numer_delete_values.shape[1]
        self.est = est
        self.numer_delete_values = numer_delete_values
        self.denom_delete_values = denom_delete_values
        
        self.pseudovalues = self.delete_values_to_pseudovalues(
            self.est, self.denom_delete_values, self.numer_delete_values
        )
        
        (self.jknife_est, self.jknife_var, self.jknife_se, self.jknife_cov) = \
            self.jknife(self.pseudovalues)

    @classmethod
    def delete_values_to_pseudovalues(cls, est, denom, numer):
        '''
        Converts delete values to pseudovalues (PyTorch batch version).

        Parameters
        ----------
        est : torch.Tensor with shape (chunk_size, )
            Whole-data ratio estimate.
        denom : torch.Tensor with shape (chunk_size, n_blocks)
            Denominator delete values.
        numer : torch.Tensor with shape (chunk_size, n_blocks)
            Numerator delete values.

        Returns
        -------
        pseudovalues : torch.Tensor with shape (chunk_size, n_blocks)
            Ratio Jackknife Pseudovalues.

        Raises
        ------
        ValueError :
            If numer.shape != denom.shape.
        '''
        if numer.shape != denom.shape:
            raise ValueError('numer.shape != denom.shape.')
        
        chunk_size, n_blocks = denom.shape
        est_expanded = est.unsqueeze(1)  # (chunk_size, 1)
        
        # 计算每个块的比值: numer / denom
        epsilon = 1e-10
        denom_safe = torch.where(
            torch.abs(denom) < epsilon,
            torch.sign(denom) * epsilon,
            denom
        )
        ratio_j = numer / denom_safe  # (chunk_size, n_blocks)
        
        # 伪值: n_blocks * est - (n_blocks - 1) * ratio_j
        pseudovalues = n_blocks * est_expanded - (n_blocks - 1) * ratio_j
        
        return pseudovalues

    @classmethod
    def jknife(cls, pseudovalues):
        '''
        Converts pseudovalues to jackknife estimate and variance (PyTorch batch version).

        Parameters
        ----------
        pseudovalues : torch.Tensor with shape (chunk_size, n_blocks)

        Returns
        -------
        jknife_est : torch.Tensor with shape (chunk_size, )
            Jackknifed estimate.
        jknife_var : torch.Tensor with shape (chunk_size, )
            Variance of jackknifed estimate.
        jknife_se : torch.Tensor with shape (chunk_size, )
            Standard error of jackknifed estimate, equal to sqrt(jknife_var).
        jknife_cov : torch.Tensor with shape (chunk_size, 1, 1)
            Covariance matrix of jackknifed estimate.
        '''
        chunk_size, n_blocks = pseudovalues.shape
        # jknife_est = pseudovalues.mean(dim=1)  # (chunk_size, )
        # jknife_est = torch.nanmean(pseudovalues, dim=1)
        mask = ~torch.isnan(pseudovalues)  # (chunk_size, n_blocks)
        count_valid = mask.sum(dim=1).float()  # (chunk_size, )
        sum_valid = torch.where(mask, pseudovalues, torch.zeros_like(pseudovalues)).sum(dim=1)  # (chunk_size, )
        # 只有在有效数量 >= 150 时才计算均值，否则设为 nan
        jknife_est = torch.where(
            count_valid >= 150,
            sum_valid / count_valid,
            torch.full_like(sum_valid, float('nan'))
        )  # (chunk_size, )

        centered = pseudovalues - jknife_est.unsqueeze(1)  # (chunk_size, n_blocks)
        # centered: (chunk_size, n_blocks)
        # jknife_cov: (chunk_size, chunk_size)
        jknife_cov = torch.matmul(centered, centered.T) / (n_blocks - 1)
        jknife_cov = jknife_cov / n_blocks
        jknife_var = torch.diagonal(jknife_cov, dim1=0, dim2=1)  # (chunk_size, )
        jknife_se = torch.sqrt(torch.clamp(jknife_var, min=0))  # (chunk_size, )
        
        return jknife_est, jknife_var, jknife_se, jknife_cov
