# -*- coding: utf-8 -#-
import numpy as np
import scipy as sp
import scipy.stats as stats
import collections

def flatten(li):
    i = 0
    while i < len(li):
        while isinstance(li[i],collections.Iterable):
            if not li[i]:
                li.pop(i)
                i -= 1
                break
            else:
                li[i:i+1] = li[i]
        i += 1
    return li

class LDA:
    def __init__(self,alpha,beta,K,docs):
        self.alpha = alpha # parameter of topics prior
        self.beta = beta # parameter of words prior
        self.K = K # num of topics 
        self.docs = docs # doc is a list of word_id

        self.M = len(docs) # num of docs
        # TODO : get the number of words
        _docs = list(docs)
        self.V = len(set(flatten(_docs))) # num of words

        self.z = [] # topics of words of docs

        self.n_d_k = np.zeros((len(self.docs),K)) # num of the topic(k) of words in the doc(d)
        self.n_k_v = np.zeros((self.V,K)) # num of the topic(k) for word(v) 
        self.n_k = np.zeros(K) # num of topic(k)

        for d,doc in enumerate(docs):
            num_of_words = len(doc)
            self.V += num_of_words
            z_d = []
            for i,v in enumerate(doc):
                z_d_i = np.random.randint(0,K)
                z_d.append(z_d_i)
                self.n_d_k[d,z_d_i] += 1
                self.n_k_v[v,z_d_i] += 1
                self.n_k[z_d_i] += 1
            self.z.append(np.array(z_d))

    # def gibbs_sampler():
    #     # initialize phi,theta,x
    #     # TODO
    #     # S: num of samples(iterations)
    #     for s in range(S):
    #         for d,doc in enumerate(self.docs):
    #             for i,v in enumerate(self,doc):
    #                 # Sample z_d_i^(s) from (3.29)
    #                 p_z = self.theta[d][k] * self.phi[k][v] / sum(theta[d]phi[:][v])
    #             # Sample theta_d^(s) from (3.36)
    #             p_theta = scipy.special.gamma() / 
    #         for k in range(K):
    #             # Sample phi_k^(s) from (3.37)
    #         # Update alpha and beta (described in Sec. 3.6)

    def collapsed_gibbs_sampler(self,S):
        # S: num of samples(iterations)
        for _ in range(S):
            for d,doc in enumerate(self.docs):
                z_d = self.z[d]
                for i,v in enumerate(doc):
                    # Update n_k_v^(s) and n_d_k^(s)
                    z_d_i = z_d[i]
                    self.n_d_k[d,z_d_i] -= 1
                    self.n_k_v[v,z_d_i] -= 1
                    self.n_k[z_d_i] -= 1

                    # Sample z_d_i^(s) from (3.38)
                    p_z = (self.n_k_v[i][:] + self.beta[v])/(self.n_k[:]+sum(self.beta[:])) *\
                            (self.n_d_k[d][:] + self.alpha[:])/(self.V + sum(self.alpha[:]))
                    new_z = np.random.multinomial(1,p_z / sum(p_z)).argmax()

                    # Update n_k_v^(s) and n_d_k^(s)
                    self.n_d_k[d,new_z] += 1
                    self.n_k_v[v,new_z] += 1
                    self.n_k[new_z] += 1
                    z_d[i] = new_z

            # Update alpha and beta (described in Sec. 3.6)
            # TODO

    def variational_bayesian_inference_single_loop(self,S=0):
        # Initialize approximation of posterior distribution
        q_z = np.array([[[1/self.K for k in range(self.K)] for _ in d] for _ in self.docs])
        q_theta = np.array([1/K for d in self.docs for _ in range(K)])
        q_phi = np.array([1/K for k in range(K) for _ in range(self.V)])

        # Derivation variational lower bound
        #TODO
        # F = np.sum(q_z*q_theta*q_phi*np.log(p_w*p_z)

        # Initialize E[n_d_k] and E[n_k_v]
        # TODO: is this correct??
        E_n_d_k = q_z.sum(axis=1)
        E_n_k_v = q_z.sum(axis=0)

        # Convergence condition 
            # A. difference between variational lower bound in each iteration
            # B. the number of iterations

        for _ in range(S):
            for d,doc in enumerate(self.docs):
                z_d = self.z[d]
                for i,v in enumerate(doc):
                    z_d_i = z_d[i]
                    # Update q(z_d_i) by (3.99)
                    q_z = {
                            np.exp(sp.psi(E_n_k_v[v]+self.beta[v])) # K-dimension
                            / np.exp(sp.psi(np.sum(E_n_k_v+np.matrix(self.beta).T,axis=0)))  # k-dimension
                            * np.exp(sp.psi(E_n_d_k[d]+self.alpha))  # K-dimension
                            / np.exp(sp.psi(np.sum(E_n_d_k+self.alpha,axis=1))) #k-dimension
                        }
                    new_q_z = q_z / sum(q_z)
                    q_z[d][i] = new_q_z
                # Update q(theta_d) by (3.90)
                #TODO: is this corrct???
                q_theta[d] = stats.dirichlet(E_n_d_k[d]+self.alpha)
            for k in range(K):
                # Update q(phi_k) by (3.96)
                # TODO: is this corrct???
                q_phi[k] = stats.dirichlet(E_n_k_v[k]+self.beta) 
            # Update alpha and beta
            #TODO

    def variational_bayesian_inference_double_loop(self,S_outer=0,S_inner=0):
        # Initialize approximation of posterior distribution
        q_z = np.array([[[1/self.K for k in range(self.K)] for _ in d] for _ in self.docs])
        q_theta = np.array([1/K for d in self.docs for _ in range(K)])
        q_phi = np.array([1/K for k in range(K) for _ in range(self.V)])

        # Derivation variational lower bound
        #TODO
        # F = np.sum(q_z*q_theta*q_phi*np.log(p_w*p_z)

        # Initialize E[n_d_k] and E[n_k_v]
        # TODO: is this correct??
        E_n_k_v = q_z.sum(axis=0)
        # Convergence criterion
            # A. difference between variational lower bound in each iteration
            # B. the number of iterations

        for _o in range(S_outer):
            for d,doc in enumerate(doc):
                n_d = len(doc)
                # Initialize E[n_d_k] = n_d/K
                E_n_d_k = np.array(len(doc) / K)
                # Absolute Error
                # TODO

                for _i in range(S_inner):
                    for i in range(n_d):
                        # Update q(z_d_i)
                        q_z = {np.exp(sp.psi(E_n_k_v[v]+self.beta[v])) # K-dimension
                                / np.exp(sp.psi(np.sum(E_n_k_v+np.matrix(self.beta).T,axis=0)))  # 1-dimension
                                * np.exp(sp.psi(E_n_d_k[d]+self.alpha))  # K-dimension
                                / np.exp(sp.psi(np.sum(E_n_d_k+self.alpha,axis=1)))} #1-dimension
                        new_q_z = q_z / sum(q_z)
                        q_z[d][i] = new_q_z
                    # Update q(theta_d) by (3.90)
                    #TODO: is this corrct???
                    q_theta[d] = stats.dirichlet(E_n_d_k[d]+self.alpha)
            for k in range(K):
                # Update q(phi_k) by (3.96)
                # TODO: is this corrct???
                q_phi[k] = stats.dirichlet(E_n_k_v[k]+self.beta) 
            #Update alpha and beta 
            # TODO
    
    def cvb0(self,):
        # Initialize q(z) randomly
        q_z = np.array([[[1/self.K for k in range(self.K)] for _ in d] for _ in self.docs]) # this is not random

        # Compute E_n_d_k, E_n_k_v by q(z)
        E_n_d_k = q_z.sum(axis=1)
        E_n_k_v = q_z.sum(axis=0)

        for Iterations:
            for d,doc in enumerate(self.doc):
                for i,v in enumerate(doc):
                    # Update q(z) by (3.131) for CVB0
                    q_z = {
                                ((E_n_k_v[v]-np.sum(q_z[d][i])) + self.beta[v])
                                / (np.sum(E_n_k_v-np.matrix(self.beta).T,axis=0))
                                * (E_n_d_k[d] + self.alpha)
                            }
                    new_q_z = q_z / sum(q_z)
                    old_q_z = q_z[d][i]
                    q_z[d][i] = new_q_z

                    for k in range(K):
                        # Update E_n_d_k = E_n_d_k - q_old(z_d_i=k) + q_new(z_d_i=k)
                        E_n_d_k[d][k] = E_n_d_k[d][k] - old_q_z[k] + new_q_z[k]

                        # If w_d_i = v, then update E_n_k_v =  E_n_k_v - q_old(z_d_i=k) + q_new(z_d_i=)
                        E_n_k_v[v][k] = E_n_k_v[v][k] - old_q_z[k] + new_q_z[k] 
                         
         # Update alpha and beta as described in Sec. 3.6
         # TODO
