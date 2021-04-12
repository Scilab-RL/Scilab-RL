import importlib
import gym

class TestingAlgos:

    base_algo_names = ['sac', 'ddpg', 'td3']
    algo_names = ['mbchac', 'her2'] + base_algo_names
    algo_names = ['mbchac']

    @staticmethod
    def get_her_performance_params(env):
        all_params = []
        if env in ['FetchReach-v1']:
            performance_params = {'n_epochs': 6, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.9, 'performance_measure': 'test/success_rate'}
        elif env in ['FetchPush-v1']:
            performance_params = {'n_epochs': 10, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.05, 'performance_measure': 'test/success_rate'}
        elif env in ['FetchSlide-v1']:
            performance_params = {'n_epochs': 50, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.03, 'performance_measure': 'test/success_rate'}
        elif env in ['FetchPickAndPlace-v1']:
            performance_params = {'n_epochs': 25, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.03, 'performance_measure': 'test/success_rate'}
        elif env in ['HandReach-v0']:
            performance_params = {'n_epochs': 70, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.1, 'performance_measure': 'test/success_rate'}
        else:
            print("Environment {} is not evaluated with HER algorithm.".format(env))
            return []
        for model in TestingAlgos.base_algo_names:
            if model in ['ppo']:
                continue
            hyper_params = {'model_class': model, 'eval_after_n_steps': 2000}
            all_params.append((performance_params, hyper_params))
        return all_params

    @staticmethod
    def get_mbchac_performance_params(env):
        all_params = []
        eval_after_n_steps = 5000
        early_stop_last_n = (10000 // eval_after_n_steps) + 1
        model = 'sac'
        hyper_params_all = {'eval_after_n_steps': eval_after_n_steps,
                            'early_stop_last_n': early_stop_last_n,
                            'plot_eval_cols': 'test/success_rate,test/mean_reward,'
                                              + 'train_0/actor_loss,train_0/critic_loss,train_0/ent_coef,train_0/n_updates,test_0/ep_success,test_0/ep_reward,train_0/ent_coef_loss,rollout_0/success_rate,'
                                              + 'train_1/actor_loss,train_1/critic_loss,train_1/ent_coef,train_1/n_updates,test_1/ep_success,test_1/ep_reward,train_1/ent_coef_loss,rollout_1/success_rate,'
                                              + 'train_2/actor_loss,train_2/critic_loss,train_2/ent_coef,train_2/n_updates,test_2/ep_success,test_2/ep_reward,train_2/ent_coef_loss,rollout_2/success_rate',
                            'render_test' : 'record',
                            'render_train': 'record',
                            'render_every_n_eval': 5,
                            'save_model_freq': 20000
                            }

        if env in ['FetchReach-v1']:
            performance_params = {'n_epochs': 60, 'n_runs': 7, 'min_success_runs': 3,
                                  'min_performance_value': 0.97, 'performance_measure': 'test/success_rate'}
        elif env in ['FetchPush-v1']:
            performance_params = {'n_epochs': 1000, 'n_runs': 3, 'min_success_runs': 1,
                                  'min_performance_value': 0.7, 'performance_measure': 'test/success_rate'}
        elif env in ['FetchSlide-v1']:
            performance_params = {'n_epochs': 1000, 'n_runs': 3, 'min_success_runs': 1,
                                  'min_performance_value': 0.7, 'performance_measure': 'test/success_rate'}
        elif env in ['FetchPickAndPlace-v1']:
            performance_params = {'n_epochs': 1000, 'n_runs': 3, 'min_success_runs': 1,
                                  'min_performance_value': 0.7, 'performance_measure': 'test/success_rate'}
        elif env in ['HandReach-v0']:
            performance_params = {'n_epochs': 1000, 'n_runs': 3, 'min_success_runs': 1,
                                  'min_performance_value': 0.7, 'performance_measure': 'test/success_rate'}
        elif env in ['HandManipulateBlock-v0']:
            performance_params = {'n_epochs': 1000, 'n_runs': 3, 'min_success_runs': 1,
                                  'min_performance_value': 0.7, 'performance_measure': 'test/success_rate'}
        elif 'Blocks-o' in env:
            performance_params = {'n_epochs': 60, 'n_runs': 7, 'min_success_runs': 3,
                                  'min_performance_value': 0.97, 'performance_measure': 'test/success_rate'}
        elif 'ButtonUnlock-o' in env:
            performance_params = {'n_epochs': 1000, 'n_runs': 3, 'min_success_runs': 1,
                                  'min_performance_value': 0.7, 'performance_measure': 'test/success_rate'}
        elif 'Hook-o' in env:
            performance_params = {'n_epochs': 1000, 'n_runs': 3, 'min_success_runs': 1,
                                  'min_performance_value': 0.7, 'performance_measure': 'test/success_rate'}
        else:
            print("Environment {} is not evaluated with HER algorithm.".format(env))
            return []

        ts = ['10,10']
        ar = [1, 0]
        sg_test_perc = [0, 0.3]
        hyper_params = {}
        for time_scales in ts:
            model_classes = [model] * len(time_scales.split(','))
            hyper_params.update({'model_classes': ",".join(model_classes), 'time_scales': time_scales})
            for action_replay in ar:
                hyper_params.update({'use_action_replay': str(action_replay)})
                for subgoal_test_perc in sg_test_perc:
                    hyper_params.update({'subgoal_test_perc': str(subgoal_test_perc)})
                    hyper_params.update(hyper_params_all)
                    all_params.append((performance_params.copy(), hyper_params.copy()))

        return all_params

    # @staticmethod
    # def get_mbchac_performance_params(env):
    #     all_params = []
    #     eval_after_n_steps = 2000
    #     early_stop_last_n = (10000 // eval_after_n_steps) + 1
    #     model = 'sac'
    #     hyper_params_all = {'eval_after_n_steps': 2000,
    #                         'early_stop_last_n': early_stop_last_n,
    #                         'plot_eval_cols': 'train/actor_loss,train/critic_loss,train/ent_coef,train/learning_rate,train/n_updates,test/success_rate,test/mean_reward,train/ent_coef_loss,rollout/success_rate'}
    #
    #     if env in ['FetchReach-v1']:
    #         performance_params = {'n_epochs': 20, 'n_runs': 7, 'min_success_runs': 4,
    #                               'min_performance_value': 0.95, 'performance_measure': 'test/success_rate'}
    #     elif env in ['FetchPush-v1']:
    #         performance_params = {'n_epochs': 10, 'n_runs': 4, 'min_success_runs': 2,
    #                               'min_performance_value': 0.05, 'performance_measure': 'test/success_rate'}
    #     elif env in ['FetchSlide-v1']:
    #         performance_params = {'n_epochs': 50, 'n_runs': 4, 'min_success_runs': 2,
    #                               'min_performance_value': 0.03, 'performance_measure': 'test/success_rate'}
    #     elif env in ['FetchPickAndPlace-v1']:
    #         performance_params = {'n_epochs': 25, 'n_runs': 4, 'min_success_runs': 2,
    #                               'min_performance_value': 0.03, 'performance_measure': 'test/success_rate'}
    #     elif env in ['HandReach-v0']:
    #         performance_params = {'n_epochs': 70, 'n_runs': 4, 'min_success_runs': 2,
    #                               'min_performance_value': 0.1, 'performance_measure': 'test/success_rate'}
    #     else:
    #         print("Environment {} is not evaluated with HER algorithm.".format(env))
    #         return []
    #
    #     for time_scales in ['_', '5,_', '2,5,_']:
    #         model_classes = [model] * len(time_scales.split(','))
    #         hyper_params = {'model_classes': ",".join(model_classes), 'time_scales': time_scales}
    #         hyper_params.update(hyper_params_all)
    #         all_params.append((performance_params.copy(), hyper_params.copy()))
    #
    #     return all_params


    @staticmethod
    def get_her2_performance_params(env):
        all_params = []
        eval_after_n_steps = 2000
        early_stop_last_n = (10000 // eval_after_n_steps) + 1
        hyper_params = {'eval_after_n_steps': eval_after_n_steps, 'early_stop_last_n': early_stop_last_n}
        if env in ['FetchReach-v1']:
            performance_params = {'n_epochs': 20, 'n_runs': 7, 'min_success_runs': 4,
                                  'min_performance_value': 0.95, 'performance_measure': 'test/success_rate'}
            hyper_params = {'eval_after_n_steps': 1000, 'train_freq': 50}
        elif env in ['FetchPush-v1']:
            performance_params = {'n_epochs': 10, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.05, 'performance_measure': 'test/success_rate'}
        elif env in ['FetchSlide-v1']:
            performance_params = {'n_epochs': 50, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.03, 'performance_measure': 'test/success_rate'}
        elif env in ['FetchPickAndPlace-v1']:
            performance_params = {'n_epochs': 25, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.03, 'performance_measure': 'test/success_rate'}
        elif env in ['HandReach-v0']:
            performance_params = {'n_epochs': 70, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.1, 'performance_measure': 'test/success_rate'}
        else:
            print("Environment {} is not evaluated with HER algorithm.".format(env))
            return []
        for model in TestingAlgos.base_algo_names:
            if model in ['ppo']:
                continue
            hyper_params_all = {'model_class': model,
                                'plot_eval_cols': 'train/actor_loss,train/critic_loss,train/ent_coef,train/learning_rate,train/n_updates,test/success_rate,test/mean_reward,train/ent_coef_loss,rollout/success_rate'}
            hyper_params.update(hyper_params_all)
            performance_params['n_epochs'] = (20000 // hyper_params['eval_after_n_steps']) + 1
            all_params.append((performance_params.copy(), hyper_params.copy()))
        return all_params

    @staticmethod
    def get_td3_performance_params(env):
        if env in []:
            performance_params = {'n_epochs': 5, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 400, 'performance_measure': 'test/mean_reward'}
            hyper_params = {'n_train_rollouts': 10}
        else:
            print("Environment {} is not evaluated with TD3 algorithm.".format(env))
            return []
        return [(performance_params, hyper_params)]

    @staticmethod
    def get_ddpg_performance_params(env):
        if env in []:
            performance_params = {'n_epochs':  5, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 400, 'performance_measure': 'test/mean_reward'}
            hyper_params = {'n_train_rollouts': 10}
        else:
            print("Environment {} is not evaluated with DDPG algorithm.".format(env))
            return []
        return [(performance_params, hyper_params)]

    @staticmethod
    def get_sac_performance_params(env):
        if env in ['MointainCarContinuous-v0']:
            performance_params = {'n_epochs':  5, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 400, 'performance_measure': 'test/mean_reward'}
            hyper_params = {'n_train_rollouts': 10}
        else:
            print("Environment {} is not evaluated with SAC algorithm.".format(env))
            return []
        return [(performance_params, hyper_params)]

    @staticmethod
    def get_dqn_performance_params(env):
        if env in ['CartPole-v1']:
            performance_params = {'n_epochs':  5, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 10.0, 'performance_measure': 'test/mean_reward'}
            hyper_params = {'n_train_rollouts': 10}
        else:
            print("Environment {} is not evaluated with DQN algorithm.".format(env))
            return []
        return [(performance_params, hyper_params)]

    # @staticmethod
    # def get_her_pytorch_performance_params(env):
    #     hyper_params = {}
    #     if env in ['AntReacherEnv-v0', 'AntCausalDepEnv-o0-v0', 'AntCausalDepEnv-o1-v0', 'AntMazeEnv-v0']:
    #         performance_params = {'epochs': 20, 'n_runs': 4, 'min_success_runs': 2,
    #                               'min_performance_value': 0.5, 'performance_measure': 'test/success_rate'}
    #     elif env == 'BlockStackMujocoEnv-gripper_random-o0-v1':
    #         performance_params = {'epochs': 4, 'n_runs': 4, 'min_success_runs': 2,
    #                               'min_performance_value': 0.7, 'performance_measure': 'test/success_rate'}
    #     elif env == 'CausalDependenciesMujocoEnv-o0-v0':
    #         performance_params = {'epochs': 4, 'n_runs': 4, 'min_success_runs': 2,
    #                               'min_performance_value': 0.7, 'performance_measure': 'test/success_rate'}
    #     else:
    #         print("Environment {} is not evaluated with HER algorithm.".format(env))
    #         return []
    #     return [(performance_params, hyper_params)]
    #
    # @staticmethod
    # def get_chac_performance_params(env):
    #     all_params = []
    #     hyper_params = {}
    #     if env == 'CausalDependenciesMujocoEnv-o0-v0':
    #         performance_params = {'epochs': 6, 'n_runs': 4, 'min_success_runs': 2, 'min_performance_value': 0.7,
    #                               'performance_measure': 'test/success_rate'}
    #         hyper_params = {'eta': 0.5, 'n_levels': 2, 'time_scales': '10,5'}
    #     elif env == 'BlockStackMujocoEnv-gripper_random-o0-v1':
    #         performance_params = {'epochs': 8, 'n_runs': 4, 'min_success_runs': 2, 'min_performance_value': 0.15,
    #                               'performance_measure': 'test/success_rate'}
    #         hyper_params = {'eta': 0.5, 'n_levels': 2, 'time_scales': '5,10'}
    #     elif env in ['AntReacherEnv-v0']:
    #         performance_params = {'epochs': 35, 'n_runs': 4, 'min_success_runs': 2, 'min_performance_value': 0.2,
    #                               'performance_measure': 'test/success_rate'}
    #         hyper_params = {'eta': 0.5, 'n_levels': 2, 'time_scales': '27,27'}
    #     else:
    #         print("Environment {} is not evaluated with CHAC algorithm.".format(env))
    #         performance_params = None
    #
    #     if performance_params is not None:
    #         all_params.append((performance_params.copy(), hyper_params.copy()))
    #
    #     return all_params
    #
    # @staticmethod
    # def get_mbchac_performance_params(env):
    #     all_params = []
    #     all_h_params = [
    #         # 1 level HER
    #         {'buffer_size': "500", 'eta': '0.0', 'time_scales': '500', 'level_types': 'her', 'simulate_level': '0'},
    #         # 2 level HAC
    #         {'buffer_size': "500,500", 'eta': '0.0,0.0', 'level_types': 'hac,hac', 'simulate_level': '0,0'},
    #         # 2 level CHAC
    #         {'buffer_size': "500,500", 'eta': '0.5,0.5', 'level_types': 'hac,hac', 'dm_hidden_size': 256,
    #          'dm_batch_size': 1024, 'dm_lr': 0.001, "dm_ensemble": 5, 'simulate_level': '0,0'},
    #         # 2 level MBCHAC
    #         {'buffer_size': "500,500", 'eta': '0.5,0.5', 'level_types': 'hac,hac', 'dm_hidden_size': 256,
    #          'dm_batch_size': 1024, 'dm_lr': 0.001, "dm_ensemble": 5, 'simulate_level': '1,1'}
    #     ]
    #     hyper_params = []
    #     if env == 'CausalDependenciesMujocoEnv-o0-v0':
    #         performance_params = [
    #             {'epochs': 4, 'min_performance_value': 0.9},
    #             {'epochs': 4, 'min_performance_value': 0.8},
    #             {'epochs': 4, 'min_performance_value': 0.8},
    #             {'epochs': 4, 'min_performance_value': 0.9, 'haltime': 500}
    #         ]
    #         hyper_params = [
    #             {'atomic_noise': 0.2, 'subgoal_noise': 0.1},
    #             {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
    #             {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
    #             {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1, 'halftime': 500},
    #         ]
    #     elif env == 'CausalDependenciesMujocoEnv-o1-v0':
    #         performance_params = [
    #             {'epochs': 25, 'min_performance_value': 0.1},
    #             {'epochs': 40, 'min_performance_value': 0.2},
    #             {'epochs': 25, 'min_performance_value': 0.2},
    #             {'epochs': 40, 'min_performance_value': 0.25},
    #         ]
    #         hyper_params = [
    #             {'atomic_noise': 0.2, 'subgoal_noise': 0.1},
    #             {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
    #             {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
    #             {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1, 'halftime': 3000}
    #         ]
    #     elif env == 'BlockStackMujocoEnv-gripper_random-o0-v1':
    #         performance_params = [
    #             {'epochs': 4, 'min_performance_value': 0.9},
    #             {'epochs': 4, 'min_performance_value': 0.3},
    #             {'epochs': 4, 'min_performance_value': 0.3},
    #             {'epochs': 4, 'min_performance_value': 0.3},
    #         ]
    #         hyper_params = [
    #             {'atomic_noise': 0.2, 'subgoal_noise': 0.1},
    #             {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
    #             {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
    #             {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1, 'halftime': 500}
    #         ]
    #     elif env in ['AntReacherEnv-v0', 'AntCausalDepEnv-o0-v0']:
    #         performance_params = [
    #             {'epochs': 20, 'min_performance_value': 0.4},
    #             {'epochs': 20, 'min_performance_value': 0.75},
    #             {'epochs': 20, 'min_performance_value': 0.7},
    #             {'epochs': 20, 'min_performance_value': 0.7},
    #         ]
    #         hyper_params = [
    #             {'atomic_noise': 0.2, 'subgoal_noise': 0.2},
    #             {'time_scales': '23,23', 'atomic_noise': 0.2, 'subgoal_noise': 0.2},
    #             {'time_scales': '23,23', 'atomic_noise': 0.2, 'subgoal_noise': 0.2},
    #             {'time_scales': '23,23', 'atomic_noise': 0.2, 'subgoal_noise': 0.2, 'halftime': 2000}
    #         ]
    #     elif env in ['AntFourRooms-v0']:
    #         performance_params = [
    #             {'epochs': 20, 'min_performance_value': 0.2},
    #             {'epochs': 20, 'min_performance_value': 0.25},
    #             {'epochs': 20, 'min_performance_value': 0.3},
    #             {'epochs': 20, 'min_performance_value': 0.35},
    #         ]
    #         hyper_params = [
    #             {'atomic_noise': 0.2, 'subgoal_noise': 0.2},
    #             {'time_scales': '27,27', 'atomic_noise': 0.2, 'subgoal_noise': 0.2},
    #             {'time_scales': '27,27', 'atomic_noise': 0.2, 'subgoal_noise': 0.2},
    #             {'time_scales': '27,27', 'atomic_noise': 0.2, 'subgoal_noise': 0.2, 'halftime': 2000}
    #         ]
    #     elif 'CopReacherEnv' in env:
    #         performance_params = [
    #             {'epochs': 10, 'min_performance_value': 0.9},
    #             {'epochs': 10, 'min_performance_value': 0.9},
    #             {'epochs': 10, 'min_performance_value': 0.95},
    #             {'epochs': 10, 'min_performance_value': 0.95},
    #         ]
    #         hyper_params = [
    #             {'atomic_noise': 0.2, 'subgoal_noise': 0.1},
    #             {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
    #             {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
    #             {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1, 'halftime': 500}
    #         ]
    #     else:
    #         print("Environment {} is not evaluated with CHAC algorithm.".format(env))
    #         performance_params = None
    #
    #     if performance_params is not None:
    #         for perf_pa, hyper_pa, all_h_p in zip(performance_params, hyper_params, all_h_params):
    #             hyper_pa.update(all_h_p)
    #             perf_pa.update({'performance_measure': 'test/success_rate', 'n_runs': 3, 'min_success_runs': 2})
    #             all_params.append((perf_pa.copy(), hyper_pa.copy()))
    #
    #     return all_params
    #
    # @staticmethod
    # def get_hiro_performance_params(env):
    #     hyper_params = {'reward_type': 'dense'}
    #     if env == 'AntMazeEnv-v0':
    #         performance_params = {'epochs': 40, 'n_runs': 2, 'min_success_runs': 1, 'min_performance_value': 0.6,
    #                               'performance_measure': 'test/success_rate'}
    #     elif env in ['CopReacherEnv-ik0-v0', 'CopReacherEnv-ik1-v0']:
    #         performance_params = {'epochs': 10, 'n_runs': 3, 'min_success_runs': 2, 'min_performance_value': 0.4,
    #                               'performance_measure': 'test/success_rate'}
    #     elif env in ['BlockStackMujocoEnv-gripper_random-o0-v1']:
    #         performance_params = {'epochs': 40, 'n_runs': 3, 'min_success_runs': 2, 'min_performance_value': 0.2,
    #                               'performance_measure': 'test/success_rate'}
    #     elif env in ['CausalDependenciesMujocoEnv-o0-v0']:
    #         performance_params = {'epochs': 40, 'n_runs': 3, 'min_success_runs': 2, 'min_performance_value': 0.4,
    #                               'performance_measure': 'test/success_rate'}
    #     else:
    #         print("Environment {} is not evaluated with HIRO algorithm.".format(env))
    #         return []
    #     return [(performance_params, hyper_params)]

    # @staticmethod
    # def get_td3_performance_params(env):
    #     hyper_params = {'reward_type': 'dense'}
    #     if env == 'CausalDependenciesMujocoEnv-o0-v0':
    #         performance_params = {'epochs': 4, 'n_runs': 4, 'min_success_runs': 2, 'min_performance_value': 0.4,
    #                               'performance_measure': 'test/success_rate'}
    #     else:
    #         print("Environment {} is not evaluated with TD3 algorithm.".format(env))
    #         return []
    #     return [(performance_params, hyper_params)]