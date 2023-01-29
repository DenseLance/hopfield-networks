import numpy as np
from PIL import Image
from typing import Optional

class ClassicalHopfield:
    """
    Based on the paper "Neural networks and physical systems with emergent collective computational abilities"
    by J. Hopfield, 1982.
    Link to paper: https://www.pnas.org/doi/10.1073/pnas.79.8.2554
    """
    def __init__(self, *, input_shape: tuple, threshold: Optional[np.ndarray] = None): # PIL.Image.size = (W, H), np.ndarray.shape = (H, W); use PIL for input_shape
        assert len(input_shape) == 2
        assert threshold is None or threshold.shape == (np.prod(input_shape),)
        
        self.input_shape = input_shape
        self.weight_shape = (np.prod(self.input_shape),) * 2
        self.weights = np.zeros(self.weight_shape)
        self.threshold = np.zeros(self.weight_shape[0]) if threshold is None else threshold # also known as bias

    def process_image(self, image: Image) -> np.ndarray:
        assert image.size == self.input_shape
        
        states = np.asarray(image.convert("1"))  # converting to black and white image
        states = states * 2 - 1 # converting image to polar values of {-1, 1}
        states = states.flatten() # converting image to a singular axis
        
        return states

    def restore_image(self, states: np.ndarray) -> Image:
        image = states.reshape(self.input_shape[::-1]) # ensure restored image is in PIL format
        image = (image + 1) // 2 * 255
        image = Image.fromarray(image).convert("1") # image stays as black and white
        
        return image

    def storage_capacity_limit(self) -> int:
        """
        Storage capacity (C) for retrieval of patterns with small percentage of errors is approximately 0.14d.
        """
        return int(0.14 * self.weight_shape[0])
    
    def energy_function(self, states: np.ndarray) -> float:
        """
        Energy should only decrease or remain constant when network is trained over time as it is negative.
        As network is updated, energy should eventually converge to a local minima.
        """
        return -np.matmul(np.matmul(self.weights, states), states) / 2 + np.matmul(self.threshold, states)

    def train(self, states: np.ndarray):
        """
        The following 2 properties should be highlighted:
        1. No node has connection with itself, therefore weight between 2 same nodes W[i][i] = 0
        -> states[i] * states [i] = 1 given states[i] ∈ {-1, 1}, hence we minus identity vector from weights
        -> Alternatively, np.fill_diagonal is used
        2. Weights are symmetric, meaning self.weights[i][j] = self.weights[j][i]
        -> Vector outer product would suffice in obtaining the weights
        """
        weights = np.outer(states, states.T)
        np.fill_diagonal(weights, 0)
        self.weights += weights

    def test(self, states: np.ndarray, synchronous: Optional[bool] = True, number_of_iterations: Optional[int] = 1) -> np.ndarray:
        """
        Two update rules can be used: synchronous and asynchronous.
        -> Synchronous: All values in state matrix are updated at the same time. Converges instantly.
        -> Asynchronous: Only one random value in state matrix is updated at any point. Converges over time.

        There is a need to set number_of_iterations if using asynchronous update rule for state to converge.
        """
        if synchronous:
            predicted_states = (np.matmul(self.weights, states) >= self.threshold) * 2 - 1 # in essence, this is dot product between weights along first axis and states, reducing axis from 2 to 1
        else:
            predicted_states = states.copy()
            for _ in range(number_of_iterations):
                index = np.random.randint(0, self.weight_shape[0])
                predicted_states[index] = (np.matmul(self.weights[index], predicted_states) >= self.threshold[index]) * 2 - 1
        
        return predicted_states
