def calculate_reward(
    action: int, 
    is_valid: bool, 
    error_state: str = None, 
    schema_adapted: bool = False,
    violation: str = None
) -> float:
    """
    Reward logic for Oversight Agent (Phase 3).
    Includes severe penalties for Ghost Allocation and Cascading Failures.
    """
    reward = 0.0
    
    # 0. Context Limit Penalties
    if error_state == "Timeout":
        return -2.0 # Forcefully drained context without deciding

    # 1. Advanced Phase 3 Violations
    if violation == "ghost_allocation":
        return -5.0 # FATAL: Approved fake patient
    elif violation == "cascading_shortage":
        return -3.0 # FATAL: Approved resource that choked downstream

    # 2. Base Rewards
    if action == 1 and is_valid:
        reward += 1.0  # True Positive
    elif action == 0 and not is_valid:
        reward += 1.0  # True Negative
    elif action == 0 and is_valid:
        reward -= 2.0  # False Negative (bad for patient care)
    elif action == 1 and not is_valid:
        reward -= 1.0  # False Positive (wastes standard money/effort)
        
    # 3. Schema Adaptation Tracking
    if schema_adapted:
        reward += 0.3
        
    # 4. Phase 2 State Penalties
    if error_state == "ProgrammingError":
        reward -= 0.2
    elif error_state == "Crash":
        reward -= 0.5
        
    return reward
