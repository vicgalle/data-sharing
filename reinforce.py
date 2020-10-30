import jax
import jax.numpy as np
from jax import grad, jit

import rlax


def run_simple_RL():

    def simple_reward(action):
        if action == 0:
            return np.array([1.])
        else:
            return np.array([0.])


    rng = jax.random.PRNGKey(0)

    grad_PG_loss = jit(grad(rlax.policy_gradient_loss))
    w_t = np.array([1.])

    d = 4
    rng, iter_rng = jax.random.split(rng)
    logits = jax.random.normal(iter_rng, shape=(1, d))

    N = 100
    for _ in range(N):
        # sample action given policy
        rng, iter_rng = jax.random.split(rng)
        a = jax.random.categorical(iter_rng, logits)
        
        # observe reward
        r = simple_reward(a)

        # update policy
        logits -= 0.1 * grad_PG_loss(logits, a, r, w_t)
        print(rlax.policy_gradient_loss(logits, a, r, w_t))

    print(logits)

def run_data_coop_game(seed, N=500):

    def data_coop_reward(a_C, a_DDO):
        if a_C == 0 and a_DDO == 0:  # both defect
            return np.array([1.]), np.array([1.])
        elif a_C == 0 and a_DDO == 1:
            return np.array([6.]), np.array([0.])
        elif a_C == 1 and a_DDO == 0:
            return np.array([0.]), np.array([6.])
        else:
            return np.array([5.]), np.array([5.])

    rng = jax.random.PRNGKey(seed)

    grad_PG_loss = jit(grad(rlax.policy_gradient_loss))
    w_t = np.array([1.])

    log = False
    d = 2
    rng, iter_rng = jax.random.split(rng)
    logits_C = 0.05*jax.random.normal(iter_rng, shape=(1, d))
    rng, iter_rng = jax.random.split(rng)
    logits_DDO = 0.05*jax.random.normal(iter_rng, shape=(1, d))

    r_Cs = []
    r_DDOs = []

    for _ in range(N):
        # sample actions given policies
        rng, iter_rng = jax.random.split(rng)
        a_C = jax.random.categorical(iter_rng, logits_C)
        rng, iter_rng = jax.random.split(rng)
        a_DDO = jax.random.categorical(iter_rng, logits_DDO)
        
        # observe rewards
        r_C, r_DDO = data_coop_reward(a_C, a_DDO)
        r_Cs.append(r_C)
        r_DDOs.append(r_DDO)

        # update policies
        logits_C -= 0.01 * grad_PG_loss(logits_C, a_C, r_C, w_t)
        logits_DDO -= 0.01 * grad_PG_loss(logits_DDO, a_DDO, r_DDO, w_t)
        
        if log:
            print('C', rlax.policy_gradient_loss(logits_C, a_C, r_C, w_t))
            print('DDO', rlax.policy_gradient_loss(logits_DDO, a_DDO, r_DDO, w_t))
            print('SU', 0.5*(r_C + r_DDO))

    print(logits_C, logits_DDO)
    return 0.5 * (np.array(r_Cs) + np.array(r_DDOs))
    


def run_data_coop_game_with_regulator(seed, N=500):

    def data_coop_reward(a_C, a_DDO):
        if a_C == 0 and a_DDO == 0:  # both defect
            return np.array([1.]), np.array([1.])
        elif a_C == 0 and a_DDO == 1:
            return np.array([6.]), np.array([0.])
        elif a_C == 1 and a_DDO == 0:
            return np.array([0.]), np.array([6.])
        else:
            return np.array([5.]), np.array([5.])

    def redistribute(r_C, r_DDO, a_R):
        tax = 0.
        if a_R == 0:
            tax = 0.
        elif a_R == 1:
            tax = 0.15
        elif a_R == 2:
            tax = 0.3
        else:
            tax = 0.5

        wealth = tax * (r_C + r_DDO)
        r_C = r_C - tax * r_C + wealth/2.
        r_DDO = r_DDO - tax * r_DDO + wealth/2.

        return r_C, r_DDO, tax
    
    def redistribute(r_C, r_DDO, a_R1, a_R2):
        tax1 = 0.
        if a_R1 == 0:
            tax1 = 0.
        elif a_R1 == 1:
            tax1 = 0.15
        elif a_R1 == 2:
            tax1 = 0.3
        else:
            tax1 = 0.5
            
        tax2 = 0.
        if a_R2 == 0:
            tax2 = 0.
        elif a_R2 == 1:
            tax2 = 0.15
        elif a_R2 == 2:
            tax2 = 0.3
        else:
            tax2 = 0.5

        wealth = tax1 * r_C + tax2 * r_DDO
        r_C = r_C - tax1 * r_C + wealth/2.
        r_DDO = r_DDO - tax2 * r_DDO + wealth/2.

        return r_C, r_DDO, tax1, tax2

    rng = jax.random.PRNGKey(seed)

    grad_PG_loss = jit(grad(rlax.policy_gradient_loss))
    w_t = np.array([1.])

    log = False
    d = 2
    rng, iter_rng = jax.random.split(rng)
    logits_C = 0.1 * np.array([[1, 1.]])
    rng, iter_rng = jax.random.split(rng)
    logits_DDO = 0.1 * np.array([[1, 1.]])
    rng, iter_rng = jax.random.split(rng)
    logits_R1 = 0.1 * np.array([[1, 1, 1, 1.]])
    logits_R2 = 0.1 * np.array([[1, 1, 1, 1.]])

    r_Cs = []
    r_DDOs = []
    taxes1 = []
    taxes2 = []

    for i in range(N):
        # sample actions given policies
        rng, iter_rng = jax.random.split(rng)
        a_C = jax.random.categorical(iter_rng, logits_C)
        rng, iter_rng = jax.random.split(rng)
        a_DDO = jax.random.categorical(iter_rng, logits_DDO)
        rng, iter_rng = jax.random.split(rng)
        a_R1 = jax.random.categorical(iter_rng, logits_R1)
        a_R2 = jax.random.categorical(iter_rng, logits_R1)
        
        # observe rewards
        r_C, r_DDO = data_coop_reward(a_C, a_DDO)
        r_Cs.append(r_C)
        r_DDOs.append(r_DDO)

        r_C, r_DDO, tax1, tax2 = redistribute(r_C, r_DDO, a_R1, a_R2)
        taxes1.append(tax1)
        taxes2.append(tax2)

        # update policies
        logits_C -= 0.01 * grad_PG_loss(logits_C, a_C, r_C, w_t)
        logits_DDO -= 0.01 * grad_PG_loss(logits_DDO, a_DDO, r_DDO, w_t)
        lag = 50
        if i % lag == 1:
            R = np.array(r_Cs[-lag:]).mean() + np.array(r_DDOs[-lag:]).mean()
            logits_R1 -= 0.005 * grad_PG_loss(logits_R1, a_R1, .5*np.array([R]), w_t)
            logits_R2 -= 0.005 * grad_PG_loss(logits_R2, a_R2, .5*np.array([R]), w_t)
        
        if log:
            print('C', rlax.policy_gradient_loss(logits_C, a_C, r_C, w_t))
            print('DDO', rlax.policy_gradient_loss(logits_DDO, a_DDO, r_DDO, w_t))
            print('SU', 0.5*(r_C + r_DDO))

    print('logits:', logits_C, logits_DDO, logits_R1, logits_R2)
    print('mean SU:', .5 * (np.mean(np.array(r_Cs)) + np.mean(np.array(r_DDOs))))
    print('mean tax1', np.array(taxes1).mean())
    print('mean tax2', np.array(taxes2).mean())

    return 0.5 * (np.array(r_Cs) + np.array(r_DDOs))

def run_data_coop_game_with_gaussian_regulator(seed, N=500):

    def data_coop_reward(a_C, a_DDO):
        if a_C == 0 and a_DDO == 0:  # both defect
            return np.array([1.]), np.array([1.])
        elif a_C == 0 and a_DDO == 1:
            return np.array([6.]), np.array([0.])
        elif a_C == 1 and a_DDO == 0:
            return np.array([0.]), np.array([6.])
        else:
            return np.array([5.]), np.array([5.])

    def gaussian_logprob(logits, a):
        return np.mean(-((a - logits)/.1)**2)

    def redistribute(r_C, r_DDO, a_R1, a_R2):
        tax1 = 0.5*jax.nn.sigmoid(a_R1)
        tax2 = 0.5*jax.nn.sigmoid(a_R2)

        wealth = tax1 * r_C + tax2 * r_DDO
        r_C = r_C - tax1 * r_C + wealth/2.
        r_DDO = r_DDO - tax2 * r_DDO + wealth/2.

        return r_C, r_DDO, tax1, tax2
    
    def redistributed(r_C, r_DDO, a_R1, a_R2):
        tax1 = 0.5*jax.nn.sigmoid(a_R1)
        tax2 = 0.5*jax.nn.sigmoid(a_R2)

        wealth = tax1 * (r_C + r_DDO)
        r_C = r_C - tax1 * r_C + wealth/2.
        r_DDO = r_DDO - tax1 * r_DDO + wealth/2.

        return r_C, r_DDO, tax1, tax2

    rng = jax.random.PRNGKey(seed)

    grad_PG_loss = jit(grad(rlax.policy_gradient_loss))
    w_t = np.array([1.])

    log = False
    d = 2
    rng, iter_rng = jax.random.split(rng)
    logits_C =  np.array([[1, 1.]])
    rng, iter_rng = jax.random.split(rng)
    logits_DDO =  np.array([[1, 1.]])
    rng, iter_rng = jax.random.split(rng)
    logits_R1 = np.array([1.])  # the mean of the Gaussian
    logits_R2 = np.array([1.])  # the mean of the Gaussian

    r_Cs = []
    r_DDOs = []
    taxes1 = []
    taxes2 = []

    for i in range(N):
        # sample actions given policies
        rng, iter_rng = jax.random.split(rng)
        a_C = jax.random.categorical(iter_rng, logits_C)
        rng, iter_rng = jax.random.split(rng)
        a_DDO = jax.random.categorical(iter_rng, logits_DDO)
        rng, iter_rng = jax.random.split(rng)
        a_R1 = 0.1*jax.random.normal(iter_rng) + logits_R1
        rng, iter_rng = jax.random.split(rng)
        a_R2 = 0.1*jax.random.normal(iter_rng) + logits_R2
        
        # observe rewards
        r_C, r_DDO = data_coop_reward(a_C, a_DDO)
        r_Cs.append(r_C)
        r_DDOs.append(r_DDO)

        r_C, r_DDO, tax1, tax2 = redistribute(r_C, r_DDO, a_R1, a_R2)
        taxes1.append(tax1)
        taxes2.append(tax2)

        # update policies
        logits_C -= 0.01 * grad_PG_loss(logits_C, a_C, r_C, w_t)
        logits_DDO -= 0.01 * grad_PG_loss(logits_DDO, a_DDO, r_DDO, w_t)
        lag = 50
        if i > 0:
            if i % lag == 0:
                R = np.array(r_Cs[-lag:]).mean() + np.array(r_DDOs[-lag:]).mean()
                logits_R1 -= 0.005 * R * grad(gaussian_logprob)(logits_R1, a_R1)
                logits_R2 -= 0.005 * R * grad(gaussian_logprob)(logits_R2, a_R2)
        
        if log:
            print('C', rlax.policy_gradient_loss(logits_C, a_C, r_C, w_t))
            print('DDO', rlax.policy_gradient_loss(logits_DDO, a_DDO, r_DDO, w_t))
            print('SU', 0.5*(r_C + r_DDO))

    print('logits:', logits_C, logits_DDO, logits_R1, logits_R2)
    print('mean SU:', .5 * (np.mean(np.array(r_Cs)) + np.mean(np.array(r_DDOs))))
    print('mean tax1', np.array(taxes1).mean())
    print('mean tax2', np.array(taxes2).mean())

    return 0.5 * (np.array(r_Cs) + np.array(r_DDOs))



