DB_NAME = 'ideas_hrl_hyperopt_database'
DB_USER = 'hyperopt_user'
DB_PW = 'Ideas21!'
DB_HOST = 'wtmpc165'
import optuna

def objective(trial):

    time_scale_values = ['50', '30', '70']
    time_scale = trial.suggest_int("time_scale", 30, 70, step=20)
    action_repla_values = [1, 0]
    action_replay = trial.suggest_categorical("action_replay", [0, 1])
    sg_test_perc_values = [0.0, 0.3]
    subgoal_test_perc = trial.suggest_float("subgoal_test_perc", 0.0, 0.3, step=0.3)
    n_succ_steps_for_early_ep_done_vales = [0, 1, 2]
    n_succ_steps_for_early_ep_done = trial.suggest_int("n_succ_steps_for_early_ep_done", 0, 3, step=1)
    n_sampled_goal_values = [2, 3, 4]
    n_sampled_goals = trial.suggest_int("n_sampled_goals", 0, 20, log=True)
    learning_rate_0 = trial.suggest_float("lr_0", 1e-5, 1e-2, log=True)
    goal_selection_strategy = trial.suggest_categorical("goal_selection_strategy", ['future'])






if __name__ == "__main__":
    # optuna.study.delete_study(study_name="ideas_hyperopt", storage=f"mysql://{DB_USER}@{DB_HOST}/{DB_NAME}")
    opt_study = optuna.create_study(study_name='ideas_hyperopt', storage=f"mysql://{DB_USER}@{DB_HOST}/{DB_NAME}", load_if_exists=True)
    opt_study.optimize(objective, n_trials=3)





