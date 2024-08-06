import itertools
import math
import random
import numpy as np
from AbsorptionMatrices.Circle import Circle
import matplotlib.pyplot as plt
from tqdm import tqdm
import functools as ft
import multiprocessing as mp

# rotate
from scipy.ndimage import rotate
from scipy.stats import norm

@ft.lru_cache(maxsize=128)
def get_p_b_i(population_size, b_i):
    return math.comb(population_size, b_i) / 2**population_size
    

@ft.lru_cache(maxsize=128)
def get_p_bi_given_A_ij_is_one(population_size, bi):
    # The probability of observing b_i given that atleast one 1 is present in the row
    if bi == 0:
        return 0
    p = 0
    # We know the row has atleast one 1, so we have to find
    # the total number of ways to get b_i-1 1s in 4 positions
    for k in range(bi):
        p += math.comb(population_size-1, k)
    return p / 2**population_size


def get_p_bi_given_A_ij_is_one_mcmc(bi, a_probs, num_samples=1000):
    # Now, a_probs is a row vector of probabilities, where one of the elements is 1
    # Now, we calculate the probability of observing bi ones in the row
    # by simulating num_samples samples from a_probs
    # and counting the number of ones in each sample
    if any(a_probs == 1) and bi == 0:
        return 0
    total_simulated = 0
    num_bi_map = {element: 0 for element in range(len(a_probs)+1)}
    for i in range(num_samples):
        # Make a binary vector
        sample = [1 if random.random() < a_probs[i] else 0 for i in range(len(a_probs))]
        num_bi = round(np.sum(sample))
        num_bi_map[num_bi] += 1
        total_simulated += 1
    #print(num_bi_map)
    num_bi_map = {k: v / total_simulated for k,v in num_bi_map.items()}
    #num_bi_map = {k: v * (1-uncertainty) + uncertainty / len(num_bi_map) for k,v in num_bi_map.items()}
    # Softmax
    d = np.sum(np.exp(list(num_bi_map.values())))
    num_bi_map = {k: np.exp(v) / d for k,v in num_bi_map.items()}
    #print(num_bi_map)
    return num_bi_map[round(bi)]

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

def update_element(j, b,a,p_b, p_a):
    a_with_j1 = np.copy(a)
    a_with_j1[j] = 1.0
    p_b_given_a = get_p_bi_given_A_i_approx(b, a_with_j1)
    p_a_given_b = p_b_given_a * p_a / p_b
    return p_a_given_b

def step(prior, measurement):
    """ The prior is the current belief of the image (N x N),
    and the measurement is a back projection of the image (1 x N).
    The measurement contains the sum of the rows of the true image.
    """
    posterior = np.zeros_like(prior)
    for i in range(prior.shape[0]):
        b = measurement[i]
        a = prior[i]
        p_b = get_p_bi_given_A_i_approx(b, a)
        # apply along the columns
        arg_gen = ((j, b, a, p_b, p_a) for j, p_a in enumerate(a))
        # calculate update_element(args) for each arg
        #with mp.Pool(10) as pool:
        #    p_a_given_b = pool.starmap(update_element, arg_gen, chunksize=3)
        posterior[i] = np.array([update_element(*arg) for arg in arg_gen])
        #posterior[i] = p_a_given_b
    return posterior

circle = Circle(60)
circle.make_holes(6, n_missing_pixels=0.4)
A_true = circle.matrix
angles = np.linspace(0, 180, 180)

# True
fig, ax = plt.subplots(1, 2)
ax[0].imshow(A_true, cmap='gray')
ax[0].set_title("True")
# Measurements
measurements = []
for angle in angles:
    rotated = rotate(A_true.copy(), angle, reshape=False, order = 5)
    measurements.append(np.sum(rotated, axis=1))
measurements = np.array(measurements)
ax[1].imshow(measurements, cmap='gray')
ax[1].set_title("Measurements")
plt.show(block=False)

prior = Circle(60).matrix - 0.2
prior = np.clip(prior, 0, 1)
#prior = np.random.rand(*A_true.shape)
fig, ax = plt.subplots(1, 2)
for measurement, angle in tqdm(zip(measurements, angles)):
    rotated_prior = rotate(prior, angle, reshape=False, order = 5)
    rotated_prior = np.clip(rotated_prior, 0, 1)
    rotated_posterior = step(rotated_prior, measurement)
    posterior = rotate(rotated_posterior, -angle, reshape=False, order = 5)
    posterior = np.clip(posterior, 0, 1)
    posterior_rounded = posterior.copy().round()
    ax[0].imshow(prior.round(2), cmap='gray')
    ax[0].set_title("Prior")
    ax[1].imshow(posterior, cmap='gray')
    ax[1].set_title("Posterior")
    prior = posterior.copy()
    plt.pause(0.1)
plt.show()