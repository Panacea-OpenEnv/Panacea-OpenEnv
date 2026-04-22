import gymnasium as gym
from gymnasium import spaces
import requests
from .reward import calculate_reward

class PanaceaEnv(gym.Env):
    """
    OpenEnv-compatible Environment for Project Panacea (Phase 3).
    Includes Trust Ledger, Decay metrics, and Context Window mapping.
    """
    
    MAX_EPISODE_STEPS = 6

    def __init__(self, api_url="http://localhost:8000"):
        super(PanaceaEnv, self).__init__()
        self.api_url = api_url
        self.current_claim = None
        
        self.action_space = spaces.Discrete(2) # 0 = Reject, 1 = Accept
        self.observation_space = spaces.Dict({
            "claim_text": spaces.Text(max_length=2000),
            "department_trust": spaces.Box(low=0.0, high=1.0, shape=(1,))
        })
        
        # Phase 3 State Tracking 
        self.schema_adapted_this_episode = False
        self.last_error = None
        self.step_count = 0
        
        # Dynamic Trust Ledger for Departments
        self.trust_ledger = {
            "Cardiology": 1.0,
            "Pulmonology": 1.0,
            "Oncology": 1.0,
            "Neurology": 1.0
        }

    def _decay_trust(self):
        """Exponentially pull trust scores towards 1.0 over time (The Boy Who Cried Wolf recovery)."""
        decay_rate = 0.1
        for dept in self.trust_ledger:
            self.trust_ledger[dept] += (1.0 - self.trust_ledger[dept]) * decay_rate

    def _penalize_trust(self, department: str, amount: float = 0.3):
        if department in self.trust_ledger:
            self.trust_ledger[department] -= amount
            if self.trust_ledger[department] < 0.0:
                self.trust_ledger[department] = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.schema_adapted_this_episode = False
        self.last_error = None
        self.step_count = 0
        
        # Decay trust positively going into a new evaluation
        self._decay_trust()
        
        try:
            response = requests.get(f"{self.api_url}/claims/pending")
            claims = response.json()
            if claims:
                self.current_claim = claims[0]
                obs = self._get_obs()
                return obs, {}
            else:
                self.current_claim = None
                return {"claim_text": "No pending claims."}, {}
        except requests.exceptions.RequestException:
            return {"claim_text": "API Connection Error"}, {}

    def step(self, action, external_metadata=None):
        """Executes a step with Phase 3 Trust and Context constraints."""
        if not self.current_claim:
            return {"claim_text": "No claims available."}, 0, True, False, {}

        self.step_count += 1
        
        # CONTEXT LIMIT EXHAUSTION (Tie-breaker/Fallback)
        if self.step_count >= self.MAX_EPISODE_STEPS:
            # Force auto-reject and heavy timeout penalty
            reward = calculate_reward(-1, False, error_state="Timeout")
            
            # Post reject to clear queue
            requests.post(f"{self.api_url}/claims/{self.current_claim['id']}/verify", json={"verdict": False})
            next_obs, _ = self.reset()
            return next_obs, reward, True, False, {"status": "timeout_exhaustion"}

        external_metadata = external_metadata or {}
        current_error = external_metadata.get('error_state')
        has_adapted = external_metadata.get('schema_adapted', False)
        
        has_schema_reward = False
        if has_adapted and self.last_error == 'ProgrammingError' and not self.schema_adapted_this_episode:
            self.schema_adapted_this_episode = True
            has_schema_reward = True

        self.last_error = current_error

        if current_error == "ProgrammingError":
            reward = calculate_reward(-1, False, error_state="ProgrammingError")
            return self._get_obs(), reward, False, False, {"status": "error_encountered"}
            
        if current_error == "Crash":
            reward = calculate_reward(-1, False, error_state="Crash")
            return self._get_obs(), reward, True, False, {"status": "crashed"}

        # Final Verification Action
        try:
            response = requests.post(
                f"{self.api_url}/claims/{self.current_claim['id']}/verify",
                json={"verdict": bool(action == 1)}
            )
            result = response.json()
            is_correct = result.get('correct', False)
            violation = result.get('violation')
            department = result.get('department')
            
            # Modify trust based on result
            if violation or not is_correct:
                if department:
                    self._penalize_trust(department)
            
            reward = calculate_reward(action, is_correct, schema_adapted=has_schema_reward, violation=violation)
            
            next_obs, _ = self.reset()
            return next_obs, reward, True, False, {"correct": is_correct, "violation": violation, "schema_adapted": has_schema_reward}
            
        except requests.exceptions.RequestException as e:
            return self._get_obs(), -0.1, True, False, {"error": str(e)}

    def render(self):
        dept = self.current_claim.get('department', 'Unknown') if self.current_claim else 'Unknown'
        trust = self.trust_ledger.get(dept, 1.0)
        print(f"-- Step {self.step_count}/{self.MAX_EPISODE_STEPS} | Dept: {dept} (Trust: {trust:.2f}) | Error: {self.last_error} --")

    def _get_obs(self):
        if self.current_claim:
            dept = self.current_claim.get('department', 'Unknown')
            trust_score = self.trust_ledger.get(dept, 1.0)
            
            text = f"Claim ID: {self.current_claim['id']}, Patient: {self.current_claim['patient_id']}, " \
                   f"Dept: {dept}, Protocol: {self.current_claim['protocol_id']}, " \
                   f"Resource: {self.current_claim['requested_resource']}, Amount: ${self.current_claim['claimed_amount']}"
                   
            return {
                "claim_text": text,
                "department_trust": [trust_score]
            }
        return {
            "claim_text": "None",
            "department_trust": [0.0]
        }
