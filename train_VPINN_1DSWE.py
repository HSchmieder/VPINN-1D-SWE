import torch
import numpy as np
import network
from scipy.special import legendre_p_all
from scipy.special import roots_legendre


class TrainModel():
    def __init__(self, device, NN, lr, weights, fct_db, Tscale, Xscale, g, cd, n_sub, n_gauss, n_test, probe_pts, probe_h):
        self.device = device
        self.VPINN = NN.to(device=self.device)
        # quadrature nodes for all subdomains
        self.n_sub = n_sub
        self.n_gauss = n_gauss
        self.n_test = n_test
        self.delta_tx = torch.tensor([1/n_sub, 2/n_sub]).reshape(1,2).to(self.device) # size subdomains
        self.gauleg_pts, self.gauleg_weights = self.get_nodes_reference()
        self.tx_pde = self.get_nodes_domain()
        # PDE based variables
        self.T = Tscale
        self.X = Xscale
        self.g = g.to(self.device)
        self.cd = cd.to(self.device)
        self.db = (fct_db(self.X*self.tx_pde[:,1])).reshape(-1,self.n_gauss**2).unsqueeze(1).to(self.device)
        # observations and respective locations scaled to [0,1]x[-1,1]
        self.tx_probe = (probe_pts/torch.tensor([self.T,self.X])).to(self.device)
        self.h_probe = probe_h.to(self.device)
        # test functions
        self.leg_poly_t, self.leg_poly_x = self.get_legendre_polynomials()
        self.phi = self.get_phi()
        self.dphidt = self.get_dphidt()
        self.dphidx = self.get_dphidx()
        # optimizer
        self.weights = weights.to(self.device)
        self.optimizer = torch.optim.Adam(self.VPINN.parameters(), lr=lr)

    def get_nodes_reference(self):
        # get quadrature nodes on reference square [-1,1]x[-1,1]
        # 1D Gauss-Legendre quadrature nodes and weights 
        pts, w = roots_legendre(self.n_gauss)
        # expand to 2D
        T, X = np.meshgrid(pts, pts)
        tx = np.stack((T.reshape(-1), X.reshape(-1)), axis=1)
        gauleg_pts = torch.from_numpy(tx).to(dtype=torch.float32, device=self.device)
        gauleg_weights = torch.from_numpy(np.outer(w,w).reshape(-1)).to(dtype=torch.float32, device=self.device)
        return gauleg_pts, gauleg_weights

    def get_nodes_domain(self):
        # get locations of all quadrature nodes in all subdomains of normalized domain [0,1]x[-1,1]
        # predimensionalize variable
        tx_tot = torch.empty(self.n_sub**2*self.n_gauss**2,2)
        # centers of subdomains
        t_center = torch.arange(self.delta_tx[0,0]/2,1,self.delta_tx[0,0])
        x_center = torch.arange(-1+self.delta_tx[0,1]/2,1,self.delta_tx[0,1])
        T_center, X_center = torch.meshgrid(t_center,x_center)
        tx_center = torch.stack((T_center.reshape(-1),X_center.reshape(-1)),dim=1).to(self.device)
        # all quadrature nodes of all subdoamins
        for i in range(self.n_sub**2):
            tx = self.gauleg_pts * 0.5*self.delta_tx + tx_center[i,:]
            tx_tot[self.n_gauss**2*i:self.n_gauss**2*(i+1),:] = tx       
        return tx_tot.to(device=self.device)

    def get_legendre_polynomials(self):
        # values of Legendre-Polynomials and 1st derivatives at reference nodes
        # legendre_p-all output format np.array [value polynomial & diff_n derivatives, n_test, n_pts]
        poly_t = legendre_p_all(self.n_test-1,self.gauleg_pts[:,0].cpu().numpy(),diff_n=1) 
        poly_x = legendre_p_all(self.n_test-1,self.gauleg_pts[:,1].cpu().numpy(),diff_n=1)
        leg_poly_t = torch.from_numpy(poly_t).to(dtype=torch.float32, device=self.device)
        leg_poly_x = torch.from_numpy(poly_x).to(dtype=torch.float32, device=self.device)
        return leg_poly_t, leg_poly_x

    def quadratic_envelope(self, pts):
        # envelope function B = (1-xi^2)/ (1-eta^2)
        # and 1st derivative
        B = (1-pts**2)
        dB = -2*pts
        return B, dB
    
    def get_phi(self):
        # phi = phit*phix = Bx(xi)*P(xi)*Bt(eta)*P(eta)
        # xi = 2*(x-x_center)/dx / eta = 2*(t-t_center)/dt 
        # values of spatial envelope function at gauleg_pts nodes
        Bx, *_ = self.quadratic_envelope(self.gauleg_pts[:,1])
        # values of temporal envelope function at gauleg_pts nodes
        Bt, *_ = self.quadratic_envelope(self.gauleg_pts[:,0])
        # values for all spatial test functions at gauleg_pts nodes 
        phix = Bx*self.leg_poly_x[0,:,:]
        # values for all temporal test functions at gauleg_pts nodes 
        phit = Bt*self.leg_poly_t[0,:,:]
        # values for all possible combinations of spatial and temporal test functions
        phi = (phit.unsqueeze(1)*phix.unsqueeze(0)).reshape(-1,self.n_gauss**2)
        return phi.to(self.device)
    
    def get_dphidt(self):
        # phi = phit*phix = Bx(xi)*P(xi)*Bt(eta)*P(eta)
        # xi = 2*(x-x_center)/dx / eta = 2*(t-t_center)/dt 
        # values of spatial envelope function at gauleg_pts nodes
        Bx, *_ = self.quadratic_envelope(self.gauleg_pts[:,1])
        # values of temporal envelope function and first derivative at gauleg_pts nodes
        Bt, dBt = self.quadratic_envelope(self.gauleg_pts[:,0])
        # values for all spatial test functions at gauleg_pts nodes
        phix = Bx*self.leg_poly_x[0,:,:]
        # values of the first derivative for all temporal test functions at gauleg_pts nodes
        # 2/self.delta_tx[0,0] results from transformation from refernece square to subdomain size
        dphit = 2/self.delta_tx[0,0]*(Bt * self.leg_poly_t[1,:,:] + dBt * self.leg_poly_t[0,:,:])
        # values for the first order derivatives for all possible combinations of spatial and temporal test functions w.r.t. time
        dphidt = (dphit.unsqueeze(1)*phix.unsqueeze(0)).reshape(-1,self.n_gauss**2)
        return dphidt.to(self.device)
    
    def get_dphidx(self):
        # phi = phit*phix = Bx(xi)*P(xi)*Bt(eta)*P(eta)
        # xi = 2*(x-x_center)/dx / eta = 2*(t-t_center)/dt 
        # values of spatial envelope function at gauleg_pts nodes
        Bx, dBx = self.quadratic_envelope(self.gauleg_pts[:,1])
        # values of temporal envelope function at gauleg_pts nodes
        Bt, *_ = self.quadratic_envelope(self.gauleg_pts[:,0])
        # values of the first derivative for all spatial test functions at gauleg_pts nodes
        # 2/self.delta_tx[0,1] results from transformation from refernece square to subdomain size
        dphix = 2/self.delta_tx[0,1]*(Bx * self.leg_poly_x[1,:,:] + dBx * self.leg_poly_x[0,:,:])
        # values for all temporal test functions at gauleg_pts nodes
        phit = Bt*self.leg_poly_t[0,:,:]
        # values for the first order derivatives for all possible combinations of spatial and temporal test functions w.r.t. space
        dphidx = (phit.unsqueeze(1)*dphix.unsqueeze(0)).reshape(-1,self.n_gauss**2)
        return dphidx.to(self.device)

    def weak_pde_residuals(self):
        # residuals w.r.t. weak 1D SWE
        # network prediction
        pred = self.VPINN.forward(self.tx_pde)
        h = pred[:,0:1].reshape(-1,self.n_gauss**2).unsqueeze(1)
        u = pred[:,1:].reshape(-1,self.n_gauss**2).unsqueeze(1)
        # conservation of mass
        unweighted_values_h = (
                         - h*self.dphidt.unsqueeze(0)
                         - h*u*self.dphidx.unsqueeze(0)*self.T/self.X
                         ).reshape(-1,self.n_gauss**2)
        res_h = unweighted_values_h @ self.gauleg_weights
        # conservation of momentum
        unweighted_values_hu = (
                         - h*u*self.dphidt.unsqueeze(0)
                         - (h*u**2+0.5*self.g*h**2)*self.dphidx.unsqueeze(0)*self.T/self.X
                         + (self.g*h*self.db + self.cd*u*torch.abs(u))*self.phi.unsqueeze(0)*self.T
                         ).reshape(-1,self.n_gauss**2)
        res_hu = unweighted_values_hu @ self.gauleg_weights
        # proper PDE scaling requires upscaling w.r.t. to integral area self.T*self.X/self.delta_tx
        # since it is only a constant factor in front of all terms it is omitted here 
        return res_h, res_hu
    
    def probe_residuals(self):
        # residuals w.r.t. to observations
        pred = self.VPINN.forward(self.tx_probe)
        h = pred[:,0]
        res = h - self.h_probe
        return res
    
    def losses(self):  
        losses = torch.zeros(3, device=self.device)
        # residuals
        res_pde1, res_pde2 = self.weak_pde_residuals()
        res_probe = self.probe_residuals()
        # losses
        losses[0] = res_pde1.pow(2).mean()
        losses[1] = res_pde2.pow(2).mean()
        losses[2] = res_probe.pow(2).mean()
        return losses
    
    def total_loss(self):
        losses = self.losses()
        losses_weighted = self.weights * losses
        return losses_weighted.sum()

    def train(self):
        loss = self.total_loss()
        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()