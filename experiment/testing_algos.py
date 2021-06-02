import importlib
import gym

class TestingAlgos:

    base_algo_names = ['sac', 'ddpg', 'td3', 'dqn']
    algo_names = ['hac', 'her'] + base_algo_names

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
        for algo in TestingAlgos.base_algo_names:
            if algo in ['ppo', 'dqn']:
                continue
            hyper_params = {'eval_after_n_steps': 2000}
            algo_params = {'model_class': algo}
            all_params.append((performance_params, hyper_params, algo_params))
        return all_params

    @staticmethod
    def get_hac_performance_params(env):
        all_params = []
        eval_after_n_steps = 5000
        early_stop_last_n = (10000 // eval_after_n_steps) + 1
        algo = 'sacvg'
        plot_col_names_template = [
            'train_##/actor_loss','train_##/critic_loss','train_##/ent_coef','train_##/n_updates',
            'test_##/ep_success','test_##/ep_reward','train_##/ent_coef_loss','rollout_##/success_rate','test_##/q_mean',
            'test_##/ep_length','train_##/ep_length','test_##/step_success'
        ]
        other_plot_col_names = ['test/success_rate', 'test/mean_reward']
        hyper_params_all = {'eval_after_n_steps': eval_after_n_steps,
                            'early_stop_last_n': early_stop_last_n,
                            'n_test_rollouts': 10,
                            'save_model_freq': 50000
                            }

        ts = [[20, 20]]
        ar = [1]
        sg_test_perc = [0.0]
        learning_rates = [[0.0035, 0.0004]]
        set_fut_ret_zero_if_done = [0]
        n_succ_steps_for_early_ep_done = [1]
        n_sampled_goal = [10]
        goal_selection_strategy = ['rndend']
        hindsight_sampling_done_if_success = [0]

        if env in ['FetchReach-v1']:
            performance_params = {'n_epochs': 80, 'n_runs': 3, 'min_success_runs': 3,
                                  'min_performance_value': 0.9, 'performance_measure': 'test/success_rate'}
            hyper_params_all.update({'eval_after_n_steps': 1000})
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
        elif 'Blocks-o0' in env:
            performance_params = {'n_epochs': 15, 'n_runs': 3, 'min_success_runs': 3,
                                  'min_performance_value': 0.9, 'performance_measure': 'test/success_rate'}
        elif 'Blocks-o1' in env:
            performance_params = {'n_epochs': 200, 'n_runs': 3, 'min_success_runs': 3,
                                  'min_performance_value': 0.9, 'performance_measure': 'test/success_rate'}
        elif 'Blocks-o' in env:
            performance_params = {'n_epochs': 100, 'n_runs': 3, 'min_success_runs': 3,
                                  'min_performance_value': 0.9, 'performance_measure': 'test/success_rate'}
        elif 'ButtonUnlock-o1' in env:
            performance_params = {'n_epochs': 60, 'n_runs': 3, 'min_success_runs': 2,
                                  'min_performance_value': 0.90, 'performance_measure': 'test/success_rate'}
            ts = [[-1, 7]]
            ar = [1]
            sg_test_perc = [0.1]
            learning_rates = [[0.0035, 0.0004]]
            set_fut_ret_zero_if_done = [0]
            n_succ_steps_for_early_ep_done = [1]
            n_sampled_goal = [7]
            goal_selection_strategy = ['rndend']
            hindsight_sampling_done_if_success = [1]
        elif 'ButtonUnlock-o2' in env:
            performance_params = {'n_epochs': 250, 'n_runs': 3, 'min_success_runs': 1,
                                  'min_performance_value': 0.97, 'performance_measure': 'test/success_rate'}
        elif 'Hook-o' in env:
            performance_params = {'n_epochs': 200, 'n_runs': 3, 'min_success_runs': 1,
                                  'min_performance_value': 0.9, 'performance_measure': 'test/success_rate'}
        elif 'AntReacher' in env or 'Ant4Rooms' in env:
            # 2 layer:
            performance_params = {'n_epochs': 100, 'n_runs': 3, 'min_success_runs': 2,
                                  'min_performance_value': 0.8, 'performance_measure': 'test/success_rate'}
            ts = [[20, 20]]
            ar = [1]
            sg_test_perc = [0.0]
            learning_rates = [[0.0035, 0.0004]]
            set_fut_ret_zero_if_done = [0]
            n_succ_steps_for_early_ep_done = [1]
            n_sampled_goal = [10]
            goal_selection_strategy = ['rndend']
            hindsight_sampling_done_if_success = [1]

        elif 'Ant' in env:
            performance_params = {'n_epochs': 100, 'n_runs': 3, 'min_success_runs': 2,
                                  'min_performance_value': 0.8, 'performance_measure': 'test/success_rate'}
            ts = [[20, 20]]
            ar = [1]
            sg_test_perc = [0.0]
            learning_rates = [[0.0035, 0.0004]]
            set_fut_ret_zero_if_done = [0]
            n_succ_steps_for_early_ep_done = [1]
            n_sampled_goal = [10]
            goal_selection_strategy = ['rndend']
            hindsight_sampling_done_if_success = [1]

        else:
            print("Environment {} is not evaluated with HER algorithm.".format(env))
            return []


        hyper_params = {}
        algo_params = {
            'render_test' : 'record',
            'render_train': 'record',
            'render_every_n_eval': 20
        }
        for frz in set_fut_ret_zero_if_done:
            algo_params.update({'set_fut_ret_zero_if_done': frz})
            for hsdis in hindsight_sampling_done_if_success:
                algo_params.update({'hindsight_sampling_done_if_success': hsdis})
                if frz != hsdis: # Setting hindsight goal transitions to done only makes sense if setting future returns to zero on done.
                    continue
                for nsg in n_sampled_goal:
                    algo_params.update({'n_sampled_goal': nsg})
                    for gss in goal_selection_strategy:
                        algo_params.update({'goal_selection_strategy': gss})
                        for time_scales in ts:
                            algo_params.update({'layer_classes': [algo] * len(time_scales)})
                            algo_params.update({'time_scales': time_scales})
                            for action_replay in ar:
                                algo_params.update({'use_action_replay': action_replay})
                                for subgoal_test_perc in sg_test_perc:
                                    algo_params.update({'subgoal_test_perc': subgoal_test_perc})
                                    for eedos in n_succ_steps_for_early_ep_done:
                                        algo_params.update({'ep_early_done_on_succ': str(eedos)})
                                        for lrs in learning_rates:
                                            n_layers = len(time_scales)
                                            if len(time_scales) != len(lrs):
                                                continue
                                            algo_params.update({'learning_rates':lrs})
                                            plot_col_names = other_plot_col_names
                                            for lay in range(n_layers):
                                                for plt_col_template in plot_col_names_template:
                                                    plot_col_names.append(plt_col_template.replace("##", str(lay)))
                                            hyper_params.update({'plot_eval_cols': plot_col_names})
                                            hyper_params.update(hyper_params_all)
                                            all_params.append((performance_params.copy(), hyper_params.copy(), algo_params.copy()))
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
        return [(performance_params, hyper_params, {})]

    @staticmethod
    def get_ddpg_performance_params(env):
        if env in []:
            performance_params = {'n_epochs':  5, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 400, 'performance_measure': 'test/mean_reward'}
            hyper_params = {'n_train_rollouts': 10}
        else:
            print("Environment {} is not evaluated with DDPG algorithm.".format(env))
            return []
        return [(performance_params, hyper_params, {})]

    @staticmethod
    def get_sac_performance_params(env):
        if env in ['MointainCarContinuous-v0']:
            performance_params = {'n_epochs':  5, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 400, 'performance_measure': 'test/mean_reward'}
            hyper_params = {'n_train_rollouts': 10}
        else:
            print("Environment {} is not evaluated with SAC algorithm.".format(env))
            return []
        return [(performance_params, hyper_params, {})]

    @staticmethod
    def get_dqn_performance_params(env):
        if env in ['CartPole-v1']:
            performance_params = {'n_epochs':  5, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 10.0, 'performance_measure': 'test/mean_reward'}
            hyper_params = {'n_train_rollouts': 10}
        else:
            print("Environment {} is not evaluated with DQN algorithm.".format(env))
            return []
        return [(performance_params, hyper_params, {})]

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