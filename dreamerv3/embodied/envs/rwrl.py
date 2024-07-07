import functools
import os

import embodied
import numpy as np


class RWRL(embodied.Env):

    DEFAULT_CAMERAS = dict(
        locom_rodent=1,
        quadruped=2,
    )

    def __init__(self, env, repeat=1, render=True, size=(64, 64), camera=-1, perturb_spec=None):
        # TODO: This env variable is meant for headless GPU machines but may fail
        # on CPU-only machines.
        if perturb_spec is None:
            perturb_spec = {}
        if "MUJOCO_GL" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"
        if isinstance(env, str):
            domain, task = env.split("_", 1)
            if camera == -1:
                camera = self.DEFAULT_CAMERAS.get(domain, 0)
            import realworldrl_suite.environments as rwrl
            env = rwrl.load(
                domain_name=domain,
                task_name=task,
                perturb_spec=perturb_spec,
                log_output='/tmp/path/to/results.npz',
                environment_kwargs=dict(log_safety_vars=True, flat_observation=True))

        self._dmenv = env
        from . import from_dm

        self._env = from_dm.FromDM(self._dmenv)
        self._env = embodied.wrappers.ExpandScalars(self._env)
        self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
        self._render = render
        self._size = size
        self._camera = camera

    @functools.cached_property
    def obs_space(self):
        spaces = self._env.obs_space.copy()
        if self._render:
            spaces["image"] = embodied.Space(np.uint8, self._size + (3,))
        return spaces

    @functools.cached_property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        for key, space in self.act_space.items():
            if not space.discrete:
                assert np.isfinite(action[key]).all(), (key, action[key])
        obs = self._env.step(action)
        if self._render:
            obs["image"] = self.render()
        return obs

    def render(self):
        return self._dmenv.physics.render(*self._size, camera_id=self._camera)
