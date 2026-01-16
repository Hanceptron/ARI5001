"""
Hidden Markov Model Implementation
ARI 5001 - Introduction to Artificial Intelligence
Theme 2: Hidden Markov Models - Filtering, Smoothing, and Sequence Decoding

Author: Murat Emirhan Aykut
Date: December 2025

This module implements the core HMM algorithms:
- Forward Algorithm (Filtering)
- Backward Algorithm
- Forward-Backward Algorithm (Smoothing)
- Viterbi Algorithm (Most Likely State Sequence)
"""

import numpy as np
from typing import Tuple, List, Optional


class HiddenMarkovModel:
    """
    Hidden Markov Model class implementing filtering, smoothing, and decoding.
    
    Mathematical Formulation:
    - States: S = {s_1, s_2, ..., s_N}
    - Observations: O = {o_1, o_2, ..., o_M}
    - Transition Matrix: A[i,j] = P(X_{t+1} = s_j | X_t = s_i)
    - Emission Matrix: B[i,k] = P(E_t = o_k | X_t = s_i)
    - Initial Distribution: π[i] = P(X_1 = s_i)
    
    References:
    - Russell & Norvig, "Artificial Intelligence: A Modern Approach", Chapter 14
    """
    
    def __init__(self, 
                 transition_matrix: np.ndarray,
                 emission_matrix: np.ndarray,
                 initial_distribution: np.ndarray,
                 state_names: Optional[List[str]] = None,
                 observation_names: Optional[List[str]] = None):
        """
        Initialize the HMM with model parameters.
        
        Args:
            transition_matrix: N x N matrix where A[i,j] = P(X_{t+1}=j | X_t=i)
            emission_matrix: N x M matrix where B[i,k] = P(E_t=k | X_t=i)
            initial_distribution: N-dimensional vector where π[i] = P(X_1=i)
            state_names: Optional list of state names
            observation_names: Optional list of observation names
        """
        self.A = np.array(transition_matrix, dtype=np.float64)
        self.B = np.array(emission_matrix, dtype=np.float64)
        self.pi = np.array(initial_distribution, dtype=np.float64)
        
        self.n_states = self.A.shape[0]
        self.n_observations = self.B.shape[1]
        
        self.state_names = state_names or [f"S{i}" for i in range(self.n_states)]
        self.observation_names = observation_names or [f"O{i}" for i in range(self.n_observations)]
        
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate that model parameters are valid probability distributions."""
        # Check transition matrix
        assert self.A.shape == (self.n_states, self.n_states), \
            f"Transition matrix must be {self.n_states}x{self.n_states}"
        assert np.allclose(self.A.sum(axis=1), 1.0), \
            "Each row of transition matrix must sum to 1"
        
        # Check emission matrix
        assert self.B.shape == (self.n_states, self.n_observations), \
            f"Emission matrix must be {self.n_states}x{self.n_observations}"
        assert np.allclose(self.B.sum(axis=1), 1.0), \
            "Each row of emission matrix must sum to 1"
        
        # Check initial distribution
        assert len(self.pi) == self.n_states, \
            f"Initial distribution must have {self.n_states} elements"
        assert np.isclose(self.pi.sum(), 1.0), \
            "Initial distribution must sum to 1"
        
        # Check non-negativity
        assert np.all(self.A >= 0), "Transition probabilities must be non-negative"
        assert np.all(self.B >= 0), "Emission probabilities must be non-negative"
        assert np.all(self.pi >= 0), "Initial probabilities must be non-negative"
    
    def forward(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward Algorithm for computing filtering distributions P(X_t | e_{1:t}).
        
        This implements the recursive computation:
        α_t(i) = P(X_t = i, e_{1:t})
        
        Base case: α_1(i) = π(i) * B(i, e_1)
        Recursion: α_{t+1}(j) = B(j, e_{t+1}) * Σ_i [α_t(i) * A(i,j)]
        
        Time Complexity: O(T * N^2) where T is sequence length, N is number of states
        Space Complexity: O(T * N) for storing all alpha values
        
        Args:
            observations: Array of observation indices of length T
            
        Returns:
            alpha: T x N matrix of forward probabilities (unnormalized)
            scaling_factors: T-dimensional array of normalization constants
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        scaling_factors = np.zeros(T)
        
        # Base case: t = 0
        alpha[0] = self.pi * self.B[:, observations[0]]
        scaling_factors[0] = alpha[0].sum()
        if scaling_factors[0] > 0:
            alpha[0] /= scaling_factors[0]
        
        # Recursion: t = 1 to T-1
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = self.B[j, observations[t]] * np.sum(alpha[t-1] * self.A[:, j])
            
            scaling_factors[t] = alpha[t].sum()
            if scaling_factors[t] > 0:
                alpha[t] /= scaling_factors[t]
        
        return alpha, scaling_factors
    
    def backward(self, observations: np.ndarray, 
                 scaling_factors: np.ndarray) -> np.ndarray:
        """
        Backward Algorithm for computing backward probabilities.
        
        This implements the recursive computation:
        β_t(i) = P(e_{t+1:T} | X_t = i)
        
        Base case: β_T(i) = 1
        Recursion: β_t(i) = Σ_j [A(i,j) * B(j, e_{t+1}) * β_{t+1}(j)]
        
        Time Complexity: O(T * N^2)
        Space Complexity: O(T * N)
        
        Args:
            observations: Array of observation indices of length T
            scaling_factors: Scaling factors from forward pass
            
        Returns:
            beta: T x N matrix of backward probabilities (scaled)
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Base case: t = T-1
        beta[T-1] = 1.0
        
        # Recursion: t = T-2 down to 0
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1]
                )
            
            # Apply same scaling as forward pass
            if scaling_factors[t+1] > 0:
                beta[t] /= scaling_factors[t+1]
        
        return beta
    
    def forward_backward(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Forward-Backward Algorithm for computing smoothed distributions P(X_t | e_{1:T}).
        
        The smoothed probability is computed as:
        γ_t(i) = P(X_t = i | e_{1:T}) = α_t(i) * β_t(i) / P(e_{1:T})
        
        Since we use scaled alpha and beta, gamma is simply:
        γ_t(i) ∝ α_t(i) * β_t(i)
        
        Time Complexity: O(T * N^2) for forward + O(T * N^2) for backward = O(T * N^2)
        Space Complexity: O(T * N)
        
        Args:
            observations: Array of observation indices of length T
            
        Returns:
            gamma: T x N matrix of smoothed probabilities
            alpha: T x N matrix of forward probabilities
            log_likelihood: Log probability of the observation sequence
        """
        # Forward pass
        alpha, scaling_factors = self.forward(observations)
        
        # Backward pass
        beta = self.backward(observations, scaling_factors)
        
        # Compute smoothed probabilities
        gamma = alpha * beta
        
        # Normalize each row
        row_sums = gamma.sum(axis=1, keepdims=True)
        gamma = np.where(row_sums > 0, gamma / row_sums, gamma)
        
        # Compute log-likelihood using scaling factors
        log_likelihood = np.sum(np.log(scaling_factors[scaling_factors > 0]))
        
        return gamma, alpha, log_likelihood
    
    def viterbi(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Viterbi Algorithm for finding the most likely state sequence.
        
        This implements dynamic programming to find:
        argmax_{x_{1:T}} P(x_{1:T} | e_{1:T})
        
        Using log probabilities:
        δ_t(j) = max_{x_{1:t-1}} [log P(x_{1:t-1}, X_t = j, e_{1:t})]
        
        Base case: δ_1(j) = log π(j) + log B(j, e_1)
        Recursion: δ_{t+1}(j) = log B(j, e_{t+1}) + max_i [δ_t(i) + log A(i,j)]
        
        Time Complexity: O(T * N^2)
        Space Complexity: O(T * N) for delta and backpointers
        
        Args:
            observations: Array of observation indices of length T
            
        Returns:
            best_path: Array of most likely state indices
            best_log_prob: Log probability of the best path
        """
        T = len(observations)
        
        # Use log probabilities to avoid underflow
        log_A = np.log(self.A + 1e-300)
        log_B = np.log(self.B + 1e-300)
        log_pi = np.log(self.pi + 1e-300)
        
        # Initialize delta (log probabilities) and backpointers
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=np.int32)
        
        # Base case
        delta[0] = log_pi + log_B[:, observations[0]]
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                # Find the best previous state
                candidates = delta[t-1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + log_B[j, observations[t]]
        
        # Backtrack to find best path
        best_path = np.zeros(T, dtype=np.int32)
        best_path[T-1] = np.argmax(delta[T-1])
        best_log_prob = delta[T-1, best_path[T-1]]
        
        for t in range(T-2, -1, -1):
            best_path[t] = psi[t+1, best_path[t+1]]
        
        return best_path, best_log_prob
    
    def filter(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute filtering distribution P(X_t | e_{1:t}) for each timestep.
        
        This is a wrapper around the forward algorithm that returns
        normalized filtering distributions.
        
        Args:
            observations: Array of observation indices
            
        Returns:
            filtered: T x N matrix where row t is P(X_t | e_{1:t})
        """
        alpha, _ = self.forward(observations)
        return alpha  # Already normalized due to scaling
    
    def smooth(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute smoothing distribution P(X_t | e_{1:T}) for each timestep.
        
        This is a wrapper around forward-backward.
        
        Args:
            observations: Array of observation indices
            
        Returns:
            smoothed: T x N matrix where row t is P(X_t | e_{1:T})
        """
        gamma, _, _ = self.forward_backward(observations)
        return gamma
    
    def generate_sequence(self, length: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a random sequence of states and observations from the HMM.
        
        Args:
            length: Length of sequence to generate
            seed: Random seed for reproducibility
            
        Returns:
            states: Array of true hidden states
            observations: Array of emitted observations
        """
        if seed is not None:
            np.random.seed(seed)
        
        states = np.zeros(length, dtype=np.int32)
        observations = np.zeros(length, dtype=np.int32)
        
        # Sample initial state
        states[0] = np.random.choice(self.n_states, p=self.pi)
        observations[0] = np.random.choice(self.n_observations, p=self.B[states[0]])
        
        # Sample subsequent states and observations
        for t in range(1, length):
            states[t] = np.random.choice(self.n_states, p=self.A[states[t-1]])
            observations[t] = np.random.choice(self.n_observations, p=self.B[states[t]])
        
        return states, observations
    
    def decode_states(self, state_indices: np.ndarray) -> List[str]:
        """Convert state indices to state names."""
        return [self.state_names[i] for i in state_indices]
    
    def decode_observations(self, obs_indices: np.ndarray) -> List[str]:
        """Convert observation indices to observation names."""
        return [self.observation_names[i] for i in obs_indices]


def create_weather_hmm(transition_noise: float = 0.0, 
                       emission_noise: float = 0.0) -> HiddenMarkovModel:
    """
    Create a weather prediction HMM.
    
    States: {Sunny (0), Rainy (1)}
    Observations: {No Umbrella (0), Umbrella (1)}
    
    Args:
        transition_noise: Amount to add to off-diagonal transitions (0 to 0.5)
        emission_noise: Amount to move emissions toward uniform (0 to 0.5)
        
    Returns:
        HMM configured for weather prediction
    """
    # Base transition matrix (weather dynamics)
    # Sunny -> Sunny: 0.8, Sunny -> Rainy: 0.2
    # Rainy -> Sunny: 0.4, Rainy -> Rainy: 0.6
    A = np.array([
        [0.8, 0.2],
        [0.4, 0.6]
    ])
    
    # Apply transition noise (make transitions less predictable)
    if transition_noise > 0:
        noise = min(transition_noise, 0.5)
        A = A * (1 - 2*noise) + noise
    
    # Base emission matrix
    # Sunny: No Umbrella 0.9, Umbrella 0.1
    # Rainy: No Umbrella 0.2, Umbrella 0.8
    B = np.array([
        [0.9, 0.1],
        [0.2, 0.8]
    ])
    
    # Apply emission noise (make observations less informative)
    if emission_noise > 0:
        noise = min(emission_noise, 0.5)
        B = B * (1 - 2*noise) + noise
    
    # Initial distribution (equal prior)
    pi = np.array([0.5, 0.5])
    
    return HiddenMarkovModel(
        transition_matrix=A,
        emission_matrix=B,
        initial_distribution=pi,
        state_names=["Sunny", "Rainy"],
        observation_names=["No Umbrella", "Umbrella"]
    )


if __name__ == "__main__":
    # Simple test
    print("Testing HMM Implementation")
    print("=" * 50)
    
    hmm = create_weather_hmm()
    
    # Generate a test sequence
    true_states, observations = hmm.generate_sequence(10, seed=42)
    
    print(f"True states: {hmm.decode_states(true_states)}")
    print(f"Observations: {hmm.decode_observations(observations)}")
    
    # Test filtering
    filtered = hmm.filter(observations)
    print(f"\nFiltering P(Sunny | e_1:t):")
    print(filtered[:, 0])
    
    # Test smoothing
    smoothed = hmm.smooth(observations)
    print(f"\nSmoothing P(Sunny | e_1:T):")
    print(smoothed[:, 0])
    
    # Test Viterbi
    best_path, log_prob = hmm.viterbi(observations)
    print(f"\nViterbi best path: {hmm.decode_states(best_path)}")
    print(f"Log probability: {log_prob:.4f}")
