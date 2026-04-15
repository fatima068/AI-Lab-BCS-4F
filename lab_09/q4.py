import numpy as np
states = ["Sunny", "Cloudy", "Rainy"]

transition_matrix = {
    "Sunny":  [0.6, 0.3, 0.1],
    "Cloudy": [0.3, 0.4, 0.3],
    "Rainy":  [0.2, 0.3, 0.5]
}

def simulate_weather(days=10):
    current = "Sunny"
    sequence = [current]

    for _ in range(days - 1):
        next_state = np.random.choice(
            states,
            p=transition_matrix[current]
        )
        sequence.append(next_state)
        current = next_state

    return sequence

# simulate 10 days
runs = 10000
rainy_counts = 0

for _ in range(runs):
    seq = simulate_weather(10)
    if seq.count("Rainy") >= 3:
        rainy_counts += 1

probability = rainy_counts / runs

print("Estimated Probability of ≥3 rainy days:", probability)
