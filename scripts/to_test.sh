#!/bin/bash

#######################################################################################################
# Testing HER-compatible algorithms and environments with sparse rewards and continuous action spaces #
#######################################################################################################

# Blocks with SAC+HER and CLEANSAC+HER
python src/main.py +performance=Blocks/o0-random-sac_her-test --multirun # OK
python src/main.py +performance=Blocks/o0-random-cleansac_her-test --multirun # OK
python src/main.py +performance=Blocks/o1-none-sac_her-test --multirun
python src/main.py +performance=Blocks/o1-none-cleansac_her-test --multirun
python src/main.py +performance=Blocks/o1-above-sac_her-test --multirun
python src/main.py +performance=Blocks/o1-above-cleansac_her-test --multirun
python src/main.py +performance=Blocks/o1-random-sac_her-test --multirun
python src/main.py +performance=Blocks/o1-random-cleansac_her-test --multirun

# ...

# FetchPickPlace with all algorithms that support HER

# HighwayParking with all algorithms that support HER.

# FetchPush with CLEANSAC+HER and SAC+HER

# FetchReach with CLEANSAC+HER and SAC+HER

# FetchSlide with CLEANSAC+HER and SAC+HER

# HighwayParking with CLEANSAC+HER and SAC+HER

# Reach1DOF with BASIC and SAC+HER

# All MetaW environments with either SAC+HER or CLEANSAC+HER


#####################################################################################
# Testing dense rewards and continuous action spaces with PPO and SAC (without HER) #
#####################################################################################


##################################################################
# Testing discrete action spaces with DQN and PPO (if supported) #
##################################################################

