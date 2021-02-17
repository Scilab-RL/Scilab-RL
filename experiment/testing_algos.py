import importlib
import gym

class TestingAlgos:
    # algorithms = ['chac', 'mbchac', 'example_algorithm', 'her_pytorch', 'hiro', 'td3']
    base_algo_names = ['ppo', 'sac', 'ddpg']
    algo_names = base_algo_names + ['her'] #'td3', 'ddpg', 'a2c',

    @staticmethod
    def get_ppo_function_testing_cmds():
        cmd = " --algorithm ppo"
        return [cmd]

    @staticmethod
    def get_sac_function_testing_cmds():
        cmd = " --algorithm sac"
        return [cmd]

    @staticmethod
    def get_td3_function_testing_cmds():
        cmd = " --algorithm td3"
        return [cmd]

    @staticmethod
    def get_ddpg_function_testing_cmds():
        cmd = " --algorithm ddpg"
        return [cmd]

    @staticmethod
    def get_a2c_function_testing_cmds():
        cmd = " --algorithm a2c"
        return [cmd]

    @staticmethod
    def get_dqn_function_testing_cmds():
        cmd = " --algorithm dqn"
        return [cmd]

    @staticmethod
    def get_her_function_testing_cmds():
        cmd = " --algorithm her"
        cmds = []
        for alg in TestingAlgos.base_algo_names:
            cmds.append( cmd + '--model_class {}'.format(alg))
        return cmds


    # def get_her_pytorch_cmds(base_cmd):
    #     cmd = base_cmd + " --algorithm baselines.her_pytorch"
    #     return [cmd]
    #
    # def get_example_algorithm_cmds(base_cmd):
    #     cmd = base_cmd + " --algorithm baselines.example_algorithm"
    #     return [cmd]
    #
    # def get_chac_cmds(base_cmd):
    #     base_cmd = base_cmd + " --algorithm baselines.chac"
    #     forward_model = [0, 1]
    #     forward_model_hs = ['32,32,32']
    #     n_levels = [1, 2]
    #     time_scales = ['10', '10,3']
    #
    #     all_cmds = []
    #     for fw in forward_model:
    #         for levels, tscale in zip(n_levels, time_scales):
    #             cmd = base_cmd
    #             cmd += " --fw {}".format(fw)
    #             cmd += " --n_levels {}".format(levels)
    #             cmd += " --time_scales {}".format(tscale)
    #             if fw:
    #                 for fwhs in forward_model_hs:
    #                     all_cmds.append(cmd + " --fw_hidden_size {}".format(fwhs))
    #             else:
    #                 all_cmds.append(cmd)
    #
    #     return all_cmds
    #
    #
    # def get_mbchac_cmds(base_cmd):
    #     base_cmd = base_cmd + " --algorithm baselines.mbchac"
    #     level_types = ['her,hac', 'hac,hac']
    #     etas = ['0.0,0.0', '0.5,0.5']
    #     simulate = ['0,0', '1,1']
    #     dm_types = ['mlp', 'rnn']
    #     dm_batch_size = ['32']
    #     dm_lr = [0.0003]
    #     ensembles = [0, 5]
    #     all_cmds = []
    #     for lvl_types in level_types:
    #         for eta in etas:
    #             for ens in ensembles:
    #                 for sim in simulate:
    #                     for dm_t, dm_b, dm_l in zip(dm_types, dm_batch_size, dm_lr):
    #                         cmd = base_cmd
    #                         cmd += " --level_types {}".format(lvl_types)
    #                         cmd += " --eta {}".format(eta)TestingAlgos.
    #                         cmd += " --dm_ensemble {}".format(ens)
    #                         cmd += " --simulate_level {}".format(sim)
    #                         cmd += " --dm_type {}".format(dm_t)
    #                         cmd += " --dm_batch_size {}".format(dm_b)
    #                         cmd += " --dm_lr {}".format(dm_l)
    #                         all_cmds.append(cmd)
    #
    #     return all_cmds
    #
    # def get_hiro_cmds(base_cmd):
    #     cmd = base_cmd + " --algorithm baselines.hiro"
    #     return [cmd]
    #
    # # def get_td3_cmds(base_cmd):
    # #     cmd = base_cmd + " --algorithm baselines.hiro --td3 1"
    # #     return [cmd]

    @staticmethod
    def get_her_performance_params(env):
        all_params = []
        hyper_params = {}
        if env in ['FetchReach-v1']:
            performance_params = {'episodes': 300, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.9, 'performance_measure': 'test/success_rate'}
        elif env in ['FetchPush-v1', 'FetchSlide-v1']:
            performance_params = {'episodes': 500, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.9, 'performance_measure': 'test/success_rate'}
        else:
            print("Environment {} is not evaluated with HER algorithm.".format(env))
            return []
        for model in TestingAlgos.base_algo_names:
            if model in ['ppo']:
                continue
            hyper_params = {'model_class': model}
            all_params.append((performance_params, hyper_params))
        return all_params

    @staticmethod
    def get_ppo_performance_params(env):
        hyper_params = {}
        if env in ['CartPole-v1']:
            performance_params = {'epochs': 5, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 400, 'performance_measure': 'test/mean_reward'}
            hyper_params = {'n_train_rollouts': 10}
        else:
            print("Environment {} is not evaluated with PPO algorithm.".format(env))
            return []
        return [(performance_params, hyper_params)]

    @staticmethod
    def get_ddpg_performance_params(env):
        hyper_params = {}
        if env in ['CartPole-v1']:
            performance_params = {'epochs': 5, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 400, 'performance_measure': 'test/mean_reward'}
            hyper_params = {'n_train_rollouts': 10}
        else:
            print("Environment {} is not evaluated with PPO algorithm.".format(env))
            return []
        return [(performance_params, hyper_params)]

    @staticmethod
    def get_sac_performance_params(env):
        hyper_params = {}
        if env in ['CartPole-v1']:
            performance_params = {'epochs': 5, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 400, 'performance_measure': 'test/mean_reward'}
            hyper_params = {'n_train_rollouts': 10}
        else:
            print("Environment {} is not evaluated with PPO algorithm.".format(env))
            return []
        return [(performance_params, hyper_params)]

    @staticmethod
    def get_dqn_performance_params(env):
        hyper_params = {}
        if env in ['CartPole-v1']:
            performance_params = {'epochs': 5, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 9.0, 'performance_measure': 'test/reward'}
            hyper_params = {'n_train_rollouts': 10}
        else:
            print("Environment {} is not evaluated with DQN algorithm.".format(env))
            return []
        return [(performance_params, hyper_params)]

    @staticmethod
    def get_her_pytorch_performance_params(env):
        hyper_params = {}
        if env in ['AntReacherEnv-v0', 'AntCausalDepEnv-o0-v0', 'AntCausalDepEnv-o1-v0', 'AntMazeEnv-v0']:
            performance_params = {'epochs': 20, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.5, 'performance_measure': 'test/success_rate'}
        elif env == 'BlockStackMujocoEnv-gripper_random-o0-v1':
            performance_params = {'epochs': 4, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.7, 'performance_measure': 'test/success_rate'}
        elif env == 'CausalDependenciesMujocoEnv-o0-v0':
            performance_params = {'epochs': 4, 'n_runs': 4, 'min_success_runs': 2,
                                  'min_performance_value': 0.7, 'performance_measure': 'test/success_rate'}
        else:
            print("Environment {} is not evaluated with HER algorithm.".format(env))
            return []
        return [(performance_params, hyper_params)]

    @staticmethod
    def get_chac_performance_params(env):
        all_params = []
        hyper_params = {}
        if env == 'CausalDependenciesMujocoEnv-o0-v0':
            performance_params = {'epochs': 6, 'n_runs': 4, 'min_success_runs': 2, 'min_performance_value': 0.7,
                                  'performance_measure': 'test/success_rate'}
            hyper_params = {'eta': 0.5, 'n_levels': 2, 'time_scales': '10,5'}
        elif env == 'BlockStackMujocoEnv-gripper_random-o0-v1':
            performance_params = {'epochs': 8, 'n_runs': 4, 'min_success_runs': 2, 'min_performance_value': 0.15,
                                  'performance_measure': 'test/success_rate'}
            hyper_params = {'eta': 0.5, 'n_levels': 2, 'time_scales': '5,10'}
        elif env in ['AntReacherEnv-v0']:
            performance_params = {'epochs': 35, 'n_runs': 4, 'min_success_runs': 2, 'min_performance_value': 0.2,
                                  'performance_measure': 'test/success_rate'}
            hyper_params = {'eta': 0.5, 'n_levels': 2, 'time_scales': '27,27'}
        else:
            print("Environment {} is not evaluated with CHAC algorithm.".format(env))
            performance_params = None

        if performance_params is not None:
            all_params.append((performance_params.copy(), hyper_params.copy()))

        return all_params

    @staticmethod
    def get_mbchac_performance_params(env):
        all_params = []
        all_h_params = [
            # 1 level HER
            {'buffer_size': "500", 'eta': '0.0', 'time_scales': '500', 'level_types': 'her', 'simulate_level': '0'},
            # 2 level HAC
            {'buffer_size': "500,500", 'eta': '0.0,0.0', 'level_types': 'hac,hac', 'simulate_level': '0,0'},
            # 2 level CHAC
            {'buffer_size': "500,500", 'eta': '0.5,0.5', 'level_types': 'hac,hac', 'dm_hidden_size': 256,
             'dm_batch_size': 1024, 'dm_lr': 0.001, "dm_ensemble": 5, 'simulate_level': '0,0'},
            # 2 level MBCHAC
            {'buffer_size': "500,500", 'eta': '0.5,0.5', 'level_types': 'hac,hac', 'dm_hidden_size': 256,
             'dm_batch_size': 1024, 'dm_lr': 0.001, "dm_ensemble": 5, 'simulate_level': '1,1'}
        ]
        hyper_params = []
        if env == 'CausalDependenciesMujocoEnv-o0-v0':
            performance_params = [
                {'epochs': 4, 'min_performance_value': 0.9},
                {'epochs': 4, 'min_performance_value': 0.8},
                {'epochs': 4, 'min_performance_value': 0.8},
                {'epochs': 4, 'min_performance_value': 0.9, 'haltime': 500}
            ]
            hyper_params = [
                {'atomic_noise': 0.2, 'subgoal_noise': 0.1},
                {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
                {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
                {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1, 'halftime': 500},
            ]
        elif env == 'CausalDependenciesMujocoEnv-o1-v0':
            performance_params = [
                {'epochs': 25, 'min_performance_value': 0.1},
                {'epochs': 40, 'min_performance_value': 0.2},
                {'epochs': 25, 'min_performance_value': 0.2},
                {'epochs': 40, 'min_performance_value': 0.25},
            ]
            hyper_params = [
                {'atomic_noise': 0.2, 'subgoal_noise': 0.1},
                {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
                {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
                {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1, 'halftime': 3000}
            ]
        elif env == 'BlockStackMujocoEnv-gripper_random-o0-v1':
            performance_params = [
                {'epochs': 4, 'min_performance_value': 0.9},
                {'epochs': 4, 'min_performance_value': 0.3},
                {'epochs': 4, 'min_performance_value': 0.3},
                {'epochs': 4, 'min_performance_value': 0.3},
            ]
            hyper_params = [
                {'atomic_noise': 0.2, 'subgoal_noise': 0.1},
                {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
                {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
                {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1, 'halftime': 500}
            ]
        elif env in ['AntReacherEnv-v0', 'AntCausalDepEnv-o0-v0']:
            performance_params = [
                {'epochs': 20, 'min_performance_value': 0.4},
                {'epochs': 20, 'min_performance_value': 0.75},
                {'epochs': 20, 'min_performance_value': 0.7},
                {'epochs': 20, 'min_performance_value': 0.7},
            ]
            hyper_params = [
                {'atomic_noise': 0.2, 'subgoal_noise': 0.2},
                {'time_scales': '23,23', 'atomic_noise': 0.2, 'subgoal_noise': 0.2},
                {'time_scales': '23,23', 'atomic_noise': 0.2, 'subgoal_noise': 0.2},
                {'time_scales': '23,23', 'atomic_noise': 0.2, 'subgoal_noise': 0.2, 'halftime': 2000}
            ]
        elif env in ['AntFourRooms-v0']:
            performance_params = [
                {'epochs': 20, 'min_performance_value': 0.2},
                {'epochs': 20, 'min_performance_value': 0.25},
                {'epochs': 20, 'min_performance_value': 0.3},
                {'epochs': 20, 'min_performance_value': 0.35},
            ]
            hyper_params = [
                {'atomic_noise': 0.2, 'subgoal_noise': 0.2},
                {'time_scales': '27,27', 'atomic_noise': 0.2, 'subgoal_noise': 0.2},
                {'time_scales': '27,27', 'atomic_noise': 0.2, 'subgoal_noise': 0.2},
                {'time_scales': '27,27', 'atomic_noise': 0.2, 'subgoal_noise': 0.2, 'halftime': 2000}
            ]
        elif 'CopReacherEnv' in env:
            performance_params = [
                {'epochs': 10, 'min_performance_value': 0.9},
                {'epochs': 10, 'min_performance_value': 0.9},
                {'epochs': 10, 'min_performance_value': 0.95},
                {'epochs': 10, 'min_performance_value': 0.95},
            ]
            hyper_params = [
                {'atomic_noise': 0.2, 'subgoal_noise': 0.1},
                {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
                {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1},
                {'time_scales': '15,15', 'atomic_noise': 0.2, 'subgoal_noise': 0.1, 'halftime': 500}
            ]
        else:
            print("Environment {} is not evaluated with CHAC algorithm.".format(env))
            performance_params = None

        if performance_params is not None:
            for perf_pa, hyper_pa, all_h_p in zip(performance_params, hyper_params, all_h_params):
                hyper_pa.update(all_h_p)
                perf_pa.update({'performance_measure': 'test/success_rate', 'n_runs': 3, 'min_success_runs': 2})
                all_params.append((perf_pa.copy(), hyper_pa.copy()))

        return all_params

    @staticmethod
    def get_hiro_performance_params(env):
        hyper_params = {'reward_type': 'dense'}
        if env == 'AntMazeEnv-v0':
            performance_params = {'epochs': 40, 'n_runs': 2, 'min_success_runs': 1, 'min_performance_value': 0.6,
                                  'performance_measure': 'test/success_rate'}
        elif env in ['CopReacherEnv-ik0-v0', 'CopReacherEnv-ik1-v0']:
            performance_params = {'epochs': 10, 'n_runs': 3, 'min_success_runs': 2, 'min_performance_value': 0.4,
                                  'performance_measure': 'test/success_rate'}
        elif env in ['BlockStackMujocoEnv-gripper_random-o0-v1']:
            performance_params = {'epochs': 40, 'n_runs': 3, 'min_success_runs': 2, 'min_performance_value': 0.2,
                                  'performance_measure': 'test/success_rate'}
        elif env in ['CausalDependenciesMujocoEnv-o0-v0']:
            performance_params = {'epochs': 40, 'n_runs': 3, 'min_success_runs': 2, 'min_performance_value': 0.4,
                                  'performance_measure': 'test/success_rate'}
        else:
            print("Environment {} is not evaluated with HIRO algorithm.".format(env))
            return []
        return [(performance_params, hyper_params)]

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