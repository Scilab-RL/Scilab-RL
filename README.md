# ideas_deep_rl2

## Set up
1. Download MuJoCo (mujoco.org) and obtain a license (as student you can obtain a free one-year student license). Copy the mjpro200_linux folder from the downloaded archive as well as mjkey.txt that you will obtain from the registration to folders of your choice
2. Set the environment variables in set_paths.sh according to the locations where you saved the mjpro200_linux folder and the mjkey.txt. If you are using an IDE, set the variables there as well.
3. Set up virtual environment using `virtualenv -p python3 venv`
4. Activate virtual environment using `source venv/bin/activate`
5. Run `./set_paths.sh`
6. Run `pip install -r requirements.txt`
7. Test your installation by running `python3 experiment/train.py --env FetchReach-v1 --algorithm her --max_try_idx 10 --n_epochs 1 --n_test_rollouts 2 --try_start_idx 3`

## Limitations
Currently, only off-policy algorithms are supported: DQN, DDPG, TD3 and SAC. PPO is not supported