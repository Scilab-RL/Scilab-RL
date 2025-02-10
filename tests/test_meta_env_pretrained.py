import unittest
import os
import yaml
import numpy as np
import src.custom_algorithms
from src.custom_algorithms.cleanppofm.cleanppofm import CLEANPPOFM
from stable_baselines3.common.env_util import make_vec_env
from src.custom_envs.moonlander.meta_env_pretrained import MetaEnvPretrained
from src.custom_envs.register_envs import register_custom_envs, register_custom_test_envs
from gymnasium.error import NameNotFound


class TestMetaEnvPretrained(unittest.TestCase):

    #@classmethod
    #def setUpClass(cls):
    #    register_custom_envs()

    def test_loading_agents_fails(self) -> None:
        register_custom_test_envs()
        with self.subTest("raise FileNotFoundError if the model with the given name does not exist"):
            with self.assertRaises(FileNotFoundError):
                env = MetaEnvPretrained("dodge_best_fm_23_09_rl_model_best", "collect_best_fm_23_08_rl_model_best")
        with self.subTest("raise FileNotFoundError if the id of the given model is not registered"):
            with self.assertRaises(NameNotFound):
                env = MetaEnvPretrained("dodge_best_fm_23_08_rl_model_best", "collect_best_fm_23_08_rl_model_best")

    def test_loading_configs(self) -> None:
        config_path_dodge_asteroids = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   "../src/custom_envs/moonlander/standard_config.yaml")
        with open(config_path_dodge_asteroids, "r") as file:
            config_dodge_asteroids = yaml.safe_load(file)

        dodge_difficulty = config_dodge_asteroids["world"]["difficulty"]
        self.assertEqual(dodge_difficulty, 'hard')

    def test_init_raised_Value_Error(self) -> None:
        register_custom_envs()
        with self.subTest(
                "raise Value error, if SoC is not in observation, but the observation should only consist of the SoC"):
            with self.assertRaises(ValueError):
                env = MetaEnvPretrained("dodge_best_fm_23_08_rl_model_best", "collect_best_fm_23_08_rl_model_best", None,
                                        None, None, True, False)
        with self.subTest(
                "raise Value error, when two reward functions are chosen (reward good switch decision & reward function of paper"):
            with self.assertRaises(ValueError):
                env = MetaEnvPretrained("dodge_best_fm_23_08_rl_model_best", "collect_best_fm_23_08_rl_model_best", None,
                                    None, None, False, False, False,
                                        False, False, False, True, True)
        with self.subTest("raise Value error, if the models have different configurations"):
            with self.assertRaises(ValueError):
                env = MetaEnvPretrained("dodge_best_fm_23_08_rl_model_best","collect_best_fm_23_08_rl_model_best",
                                        config_file_name_dodge_asteroids='dodge_test_config.yaml', config_file_name_collect_asteroids='collect_test_config.yaml')
            raise NotImplementedError

    def test_init_everything_set_to_their_initial_value(self) -> None:
        register_custom_envs()
        env = MetaEnvPretrained("dodge_best_fm_23_08_rl_model_best", "collect_best_fm_23_08_rl_model_best")
        self.assertEqual(env.SoC_dodge, np.array([1.0]))
        self.assertEqual(env.SoC_collect, np.array([1.0]))
        self.assertEqual(env.prediction_error_dodge, -1)
        self.assertEqual(env.prediction_error_collect, -1)
        self.assertEqual(env.need_for_control_dodge, -1)
        self.assertEqual(env.need_for_control_collect, -1)



    if __name__ == '__main__':
        unittest.main()