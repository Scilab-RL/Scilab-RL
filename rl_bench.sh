DetectedConditionEnvs="close_grill
reach_target
place_shape_in_shape_sorter
put_rubbish_in_bin
take_umbrella_out_of_umbrella_stand
empty_dishwasher
sweep_to_dustpan
straighten_rope
water_plants
take_item_out_of_drawer
put_item_in_drawer
slide_block_to_target
open_grill"
# close_grill is very slow
# empty_dishwasher, take_item_out_of_drawer and put_item_in_drawer are also rather slow
# for some reason, open_grill is not slow.

for ENV in $DetectedConditionEnvs
do
  ENV="$ENV-state-v0"
  python experiment/train.py env=$ENV algorithm=hac layer_classes=['sac','ddpg'] algorithm.render_train=display algorithm.time_scales=[5,-1] n_epochs=1 eval_after_n_steps=100 n_test_rollouts=1
done
