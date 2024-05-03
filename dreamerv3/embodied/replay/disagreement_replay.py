import numpy as np
from dreamerv3.embodied.replay.base_prioritized_reverb import BasePrioritizedReverb
from dreamerv3.expl import Disag


class DisagreementReplay(BasePrioritizedReverb):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_agent(self, agent):
        super().set_agent(agent)
        # TODO: need to use the agent's exploration disag if it already exists
        # self.disag = Disag(
        #     agent.wm, agent.act_space, agent.config, name="priority_disag"
        # )
        # self.should_train_disag = True

    def train(self, data):
        if self.should_train_disag:
            return self.disag.train(data)
        else:
            return super().train(data)

    @staticmethod
    def _calculate_priority_score(disag, hyper):
        # TODO: remove this function after updating prioritize
        return (hyper["c"] * disag)

    def prioritize(self, key, env_steps, losses, td_error):
        # Could potentially have the disag_score be passed into here instead of manually calculated everytime.
        # TODO: need to switch this into using the output of the disag model
        flat_steps = (
            env_steps.flatten()
        )  # Are these the states? If so can be pass then to disag
        flat_losses = losses.flatten()
        flat_count = self.visit_count[flat_steps]
        flat_priority = (
            self._calculate_priority_score(flat_losses, flat_count, self.hyper)
            / self.priority_scalar
        )
        flat_keys = self._combine_key(
            self.step_to_keyA[flat_steps], self.step_to_keyB[flat_steps]
        )
        flat_updates = {int(k): p for k, p in zip(flat_keys, flat_priority)}
        self.client.mutate_priorities("table", flat_updates)
