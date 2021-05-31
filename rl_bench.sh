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
# for some reason, open_grill is not slow

DetectedAndNothingGraspedConditionEnvs="take_cup_out_from_cabinet
setup_checkers
play_jenga
light_bulb_out
meat_on_grill
open_jar
put_groceries_in_cupboard
place_cups
reach_and_drag
take_shoes_out_of_box
take_toilet_roll_off_stand
move_hanger
take_money_out_safe
light_bulb_in
phone_on_base
take_usb_out_of_computer
put_shoes_in_box
take_off_weighing_scales
weighing_scales
insert_usb_in_computer
beat_the_buzz
stack_wine
put_money_in_safe
put_bottle_in_fridge
close_jar
hang_frame_on_hanger
stack_cups
put_umbrella_in_umbrella_stand
take_frame_off_hanger
solve_puzzle
remove_cups
set_the_table
open_oven
place_hanger_on_rack
plug_charger_in_power_supply
put_books_on_bookshelf
take_tray_out_of_oven
meat_off_grill
take_plate_off_colored_dish_rack
unplug_charger
open_wine_bottle
change_clock
put_tray_in_oven
put_toilet_roll_on_stand"
# place_cups, remove_cups fail because they add a condition each time reset() is called
# put_books_on_bookshelf is rather slow

for ENV in $DetectedAndNothingGraspedConditionEnvs
do
  ENV="$ENV-state-v0"
  python experiment/train.py env=$ENV algorithm=hac layer_classes=['sac','ddpg'] algorithm.render_train=display algorithm.time_scales=[5,-1] n_epochs=1 eval_after_n_steps=100 n_test_rollouts=1
done
