

def calculate_reward(speed_diff):
    reward = -speed_diff  # Penalize deviations from the target speed

    total_reward = reward

    return total_reward