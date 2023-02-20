import numpy as np
from scipy.stats import multivariate_normal

# 求x的概率
# 假设x服从高斯分布
import math
def gaussian_probability(x,mu0,sigma0):
    # leftpart=1/(2*math.pi*sigma0)
    # exp_part=-1*np.dot(x-mu0,x-mu0)/(2*sigma0)
    # p=leftpart*math.exp(exp_part)
    p=0
    try:
        p=multivariate_normal.pdf(x,mu0,sigma0*np.eye(mu0.shape[0]))
    except:
        print(x,mu0,sigma0*np.eye(mu0.shape[0]))
    return p


def e_step(X, pi, A, mu, sigma2):
    """E-step: forward-backward message passing"""
    # Messages and sufficient statistics
    N, T, K = X.shape
    M = A.shape[0]
    alpha = np.zeros([N,T,M])  # [N,T,M]
    alpha_sum = np.zeros([N,T])  # [N,T], normalizer for alpha
    beta = np.zeros([N,T,M])  # [N,T,M]
    gamma = np.zeros([N,T,M])  # [N,T,M]
    xi = np.zeros([N,T-1,M,M])  # [N,T-1,M,M]
    # print(mu)
    # print(sigma2)
    # Forward messages
    for i in range(N):
        for zi in range(M):
            # print(X[i,0],mu[zi],sigma2[zi])
            alpha[i,0,zi]=pi[zi]*gaussian_probability(X[i,0],mu[zi,:],sigma2[zi])
            alpha_sum[i,0]=alpha_sum[i,0]+alpha[i,0,zi]
        
        for zi in range(M):
            alpha[i,0,zi]=alpha[i,0,zi]/alpha_sum[i,0]
        # 上一步没有问题
        
        for t in range(1,T):
            for zt in range(M):
                sum_p_alpha=0
                for zt_1 in range(M):
                    sum_p_alpha=sum_p_alpha+(A[zt_1,zt]*alpha[i,t-1,zt_1])
                alpha[i,t,zt]=gaussian_probability(X[i,t],mu[zt],sigma2[zt])*sum_p_alpha
                alpha_sum[i,t]=alpha_sum[i,t]+alpha[i,t,zt]
                
            for zt in range(M):
                alpha[i,t,zt]=alpha[i,t,zt]/alpha_sum[i,t]
        
    # Backward messages
    # TODO ...
        for zi in range(M):
            beta[i,T-1,zi]=1
        
        for t in range(T-1,0,-1):
            for zt_1 in range(M):
                beta_sum=0
                for zt in range(M):
                    beta_sum=beta_sum+A[zt_1,zt]*gaussian_probability(X[i,t]
                    ,mu[zt],sigma2[zt])*beta[i,t,zt]
                beta[i,t-1,zt_1]=1/alpha_sum[i,t]*beta_sum
        
        # 问题出在xi的计算上

        for t in range(T):
            for zt in range(M):
                gamma[i,t,zt]=alpha[i,t,zt]*beta[i,t,zt]
        
        for t in range(1,T):
            for zt_1 in range(M):
                for zt in range(M):
                    xi[i,t-1,zt_1,zt]=(1/alpha_sum[i,t])*A[zt_1,zt]*gaussian_probability(X[i,t],
                    mu[zt],sigma2[zt])*alpha[i,t-1,zt_1]*beta[i,t,zt]
    
    # Sufficient statistics
    # TODO ...

    # Although some of them will not be used in the M-step, please still
    # return everything as they will be used in test cases
    return alpha, alpha_sum, beta, gamma, xi


def m_step(X, gamma, xi):
    """M-step: MLE"""
    # TODO ...
    N, T, K = X.shape
    M = gamma.shape[2]
    # N T M
    pi = np.zeros(M)
    # 4中状态
    A = np.zeros([M,M])
    mu=np.zeros([M,K])
    # 不同的状态对应不同的分布
    sigma2=np.zeros(M)
    for i in range(M):
        for x in range(N):
            pi[i]=pi[i]+gamma[x,0,i]
        pi[i]=pi[i]/N
    
    pi = pi / pi.sum()
    
    # A[i,j]计算有问题
    # sum(A[i,:])应该等于1
    # 这里不等于1

    for i in range(M):
        for j in range(M):
            for x in range(N):
                one_x_sum=0
                for t in range(1,T):
                    one_x_sum=one_x_sum+xi[x,t-1,i,j]
                A[i,j]=A[i,j]+one_x_sum
            A[i,j]=A[i,j]/N
    
    A = A / A.sum(axis=-1, keepdims=True)
    
    for i in range(M):

        expect_upper_value=0
        expect_lower_value=0
        for x in range(N):
            upper_value=0
            lower_value=0
            for t in range(T):
                upper_value=upper_value+gamma[x,t,i]*X[x,t,:]
                lower_value=lower_value+gamma[x,t,i]
            expect_upper_value=expect_upper_value+upper_value
            expect_lower_value=expect_lower_value+lower_value
        
        mu[i]=expect_upper_value/expect_lower_value
        
    for i in range(M):
        
        expect_lower_value=0
        expect_upper_value=0
        
        for x in range(N):
            upper_value=0
            lower_value=0
            for t in range(T):
                upper_value=upper_value+gamma[x,t,i]*np.square(np.linalg.norm(X[x,t,:]-mu[i]))
                lower_value=lower_value+K*gamma[x,t,i]
            expect_upper_value=expect_upper_value+upper_value
            expect_lower_value=expect_lower_value+lower_value
        sigma2[i]=expect_upper_value/expect_lower_value

    return pi, A, mu, sigma2

def hmm_train(X, pi, A, mu, sigma2, em_step=1):
    """Run Baum-Welch algorithm."""
    # 执行E M 算法
    
    for step in range(em_step):

        alpha, alpha_sum, beta, gamma, xi = e_step(X, pi, A, mu, sigma2)
        print(mu,sigma2)
        print(alpha, alpha_sum, beta, gamma, xi)
        
        
        # 在max 的时候可能存在问题
        
        pi, A, mu, sigma2 = m_step(X, gamma, xi)
        # 在m步会更新 概率分布的参数
        # pi, A, mu, sigma2 
        print(pi, A, mu, sigma2)
        
        print(f"step: {step}  ln p(x): {np.einsum('nt->', np.log(alpha_sum))}")
    return pi, A, mu, sigma2


def hmm_generate_samples(N, T, pi, A, mu, sigma2):
    """Given pi, A, mu, sigma2, generate [N,T,K] samples."""
    M, K = mu.shape
    # M*K 是 mu的 shape
    # 4*2
    # Y是隐变量

    Y = np.zeros([N,T], dtype=int) 
    X = np.zeros([N,T,K], dtype=float)
    for n in range(N):
        Y[n,0] = np.random.choice(M, p=pi)  # [1,]
        X[n,0,:] = multivariate_normal.rvs(mu[Y[n,0],:], sigma2[Y[n,0]] * np.eye(K))  # [K,]
    for t in range(T - 1):
        for n in range(N):
            Y[n,t+1] = np.random.choice(M, p=A[Y[n,t],:])  # [1,]
            X[n,t+1,:] = multivariate_normal.rvs(mu[Y[n,t+1],:], sigma2[Y[n,t+1]] * np.eye(K))  # [K,]
    return X


def main():
    """Run Baum-Welch on a simulated toy problem."""
    # Generate a toy problem
    np.random.seed(12345)  # for reproducibility
    # M表示离散的状态 4中状态 隐变量
    # K是
    # T是步数 100步
    # K 是 x的维度
    # z的维度
    N, T, M, K = 10, 100, 4, 2
    # 概率

    pi = np.array([.0, .0, .0, 1.])  # [M,]
    # 状态转移矩阵
    A = np.array([[.7, .1, .1, .1],
                  [.1, .7, .1, .1],
                  [.1, .1, .7, .1],
                  [.1, .1, .1, .7]])  # [M,M]
    
    mu = np.array([[2., 2.],
                   [-2., 2.],
                   [-2., -2.],
                   [2., -2.]])  # [M,K]
    
    sigma2 = np.array([.2, .4, .6, .8])  # [M,]
    X = hmm_generate_samples(N, T, pi, A, mu, sigma2)

    # Run on the toy problem
    pi_init = np.random.rand(M)
    pi_init = pi_init / pi_init.sum()
    # pi_init 是四个状态的概率

    A_init = np.random.rand(M, M)
    # A_init 是四种状态之间的转移矩阵
    A_init = A_init / A_init.sum(axis=-1, keepdims=True)
    mu_init = 2 * np.random.rand(M, K) - 1
    # 初始化mu矩阵 4*2


    sigma2_init = np.ones(M)

    pi, A, mu, sigma2 = hmm_train(X, pi_init, A_init, mu_init, sigma2_init, em_step=20)
    print(pi)
    print(A)
    print(mu)
    print(sigma2)

if __name__ == '__main__':
    main()
