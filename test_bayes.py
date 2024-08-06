import itertools
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Now each element (i,j) in B1 and B2 is the observation for the i-th row and j-th column
# Similarly, each element b1_i is the observation for the i-th row in A
# and each element b2_j is the observation for the j-th column in A

# We want to update the belief P_curr to P_next using the observations B1 and B2
# We will use the Bayes rule to update the belief
# P_next_ij = P(b_i|A_ij)*P(A_ij) / P(b_i)
# P(A)_ij = P_curr_ij
# The probability of observing b_i is simply the probability of observing b_i over all A_i
# P(b_i) = ncr(5, b_i) / (2^5)
# The probability P(b_i|A_ij) is the probability of observing b_i given A_ij is 1
# For example, if b_i = 0, then P(b_i|A_ij) = 0, since if A_ij=1, the row i has atleast one 1.
# On the otherhand, if b_i = 5, then P(b_i|A_ij) = 1, since the row is full of 1s anyway
# The probability of observing b_i given A_ij=1 can be calculated as:
# P(b_i|A_ij) = sum_{k=b_i}^{5} ncr(5, k) / 2^5
# Let's update the belief P_curr to P_next using the observations B1 and B2

def get_p_b_i(b_i):
    return math.comb(5, b_i) / 2**5
    

def get_p_bi_given_A_ij_is_one(bi):
    # The probability of observing b_i given that atleast one 1 is present in the row
    if bi == 0:
        return 0
    p = 0
    # We know the row has atleast one 1, so we have to find
    # the total number of ways to get b_i-1 1s in 4 positions
    for k in range(bi):
        p += math.comb(4, k)
    return p / 2**4


def get_p_bi_given_A_ij_is_one_mcmc(bi, a_probs, num_samples=10000):
    # Now, a_probs is a row vector of probabilities, where one of the elements is 1
    # Now, we calculate the probability of observing bi ones in the row
    # by simulating num_samples samples from a_probs
    # and counting the number of ones in each sample
    if any(a_probs == 1) and bi == 0:
        return 0
    total_simulated = 0
    num_bi_map = {element: 0 for element in range(6)}
    for i in range(num_samples):
        # Make a binary vector
        sample = [1 if random.random() < a_probs[i] else 0 for i in range(5)]
        num_bi = np.sum(sample)
        num_bi_map[num_bi] += 1
        total_simulated += 1
    print(num_bi_map)
    num_bi_map = {k: v / total_simulated for k,v in num_bi_map.items()}
    # Softmax
    #d = np.sum(np.exp(list(num_bi_map.values())))
    #num_bi_map = {k: np.exp(v) / d for k,v in num_bi_map.items()}
    #print(num_bi_map)
    return num_bi_map[bi]


def get_p_bi_given_A_ij_is_one_analytical(bi, a_probs):
    n = len(a_probs)
    if bi == 0 and 1 in a_probs:
        return 0
    
    # Initialize the probability
    prob = 0
    
    # Generate all subsets of size bi
    for subset in itertools.combinations(range(n), round(bi)):
        # Calculate the product of p_i for i in subset
        p_product = 1
        for i in subset:
            p_product *= a_probs[i]
        # Calculate the product of (1 - p_j) for j not in subset
        for j in range(n):
            if j not in subset:
                p_product *= (1 - a_probs[j])
        # Sum the probabilities
        prob += p_product
    
    return prob

def get_p_bi_given_A_i_approx(bi, a_probs):
    # Calculate the mean (mu) and variance (sigma^2)
    #bi = round(bi)
    mu = np.sum(a_probs)
    sigma2 = np.sum(a_probs * (1 - a_probs))
    
    if bi == 0 and 1 in a_probs:
        return 0
    
    # Standard deviation (sigma)
    sigma = np.sqrt(sigma2)
    
    # Use the normal approximation
    probability = norm.pdf(bi, loc=mu, scale=sigma)
    #probability = np.clip(probability, 0, 1)
    return probability
    

# The unknown truth
A_true = np.array([
    [1,1,0,0,0],
    [0,1,0,0,0],
    [0,1,0,0,0],
    [0,0,0,0,0],
    [0,0,0,0,0]
])

# The current belief
P_curr = np.array([
    [0.2,0.2,0.2,0.2,0.2],
    [0.2,0.2,0.2,0.2,0.2],
    [0.2,0.2,0.2,0.2,0.2],
    [0.2,0.2,0.2,0.2,0.2],
    [0.2,0.2,0.2,0.2,0.2]
]) + 0.6

P_curr = np.random.rand(5,5)

# Projections
b1 = np.sum(A_true, axis=1)
b2 = np.sum(A_true, axis=0)

# repeat scan_1 -> 5 columns
B1 = np.array([b1,b1,b1,b1,b1])
# repeat scan_2 -> 5 rows
B2 = np.array([b2,b2,b2,b2,b2])
print(B1)
print(B2)

# Plot p_bi_given_A_ij_is_one
fig, ax = plt.subplots()
probs = []
for i in range(5):
    p = get_p_bi_given_A_ij_is_one_analytical(i, np.array([1,0.5,0.5,0.5,0.5]))
    probs.append(p)
ax.bar(list(range(5)), probs)
ax.set_title("Probability of observing 'b' given atleast one 1 is present in the row")
ax.set_xlabel("Measurement (b)")
ax.set_ylabel("Probability")

# plot prob_bi
fig, ax = plt.subplots()
probs = []
for i in range(6):
    p = get_p_b_i(i)
    probs.append(p)
ax.bar(list(range(6)), probs)
ax.set_title("Probability of observing 'b'")
ax.set_xlabel("Measurement (b)")
ax.set_ylabel("Probability")

plt.show(block=False)

# Calculate a posterior
P_next = np.zeros((5,5))
for i in range(5):
    row = i
    b = b1[row]
    a = P_curr[i,:]
    for j in range(5):
        a_with_j1 = np.copy(a)
        a_with_j1[j] = 1
        p_b_given_a = get_p_bi_given_A_i_approx(b, a_with_j1)
        print(f"Probability of observing {b} given {a_with_j1} is {p_b_given_a}")
        p_b = get_p_bi_given_A_i_approx(b, a)
        p_a = P_curr[i,j]
        p_a_given_b = p_b_given_a * p_a / p_b
        P_next[i,j] = p_a_given_b
print(P_next)
P_curr = P_next.copy()

P_next = np.zeros((5,5))
for i in range(5):
    col = i
    b = b2[col]
    a = P_curr[:,i]
    for j in range(5):
        a_with_j1 = np.copy(a)
        a_with_j1[j] = 1
        p_b_given_a = get_p_bi_given_A_i_approx(b, a_with_j1)
        print(f"Probability of observing {b} is {p_b_given_a}")
        p_b = get_p_bi_given_A_i_approx(b, a)
        p_a = P_curr[j,i]
        p_a_given_b = p_b_given_a * p_a / p_b
        P_next[j,i] = p_a_given_b
print(P_next)
        






    








