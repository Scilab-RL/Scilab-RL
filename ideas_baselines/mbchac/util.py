import numpy as np

def compute_time_scales(time_scales_str, env):
    scales = time_scales_str.split(",")
    max_steps = env.spec.max_episode_steps
    for i,s in enumerate(scales):
        if s == '_':
            defined_steps = np.product([int(step) for step in scales[:i]])
            defined_after_steps = np.product([int(step) for step in scales[i+1:]])
            defined_steps *= defined_after_steps
            assert max_steps % defined_steps == 0, "Error defined time_scale not compatible with environment max steps. Max. number of environment steps {} needs to be divisible by product of all defined steps {}.".format(max_steps, defined_steps)
            this_steps = int(max_steps / defined_steps)
            scales[i] = str(this_steps)
    assert np.product([int(step) for step in scales]) == max_steps, "Error defined time_scale not compatible with environment max steps. Product of all steps needs to be {}".format(max_steps)
    return ",".join(scales)

def get_concat_dict_from_dict_list(dict_list):
    concat_info = {}
    for inf in dict_list:
        for k,v in inf.items():
            if k not in concat_info:
                concat_info[k] = []
            if type(v) == list:
                concat_info[k] += v
            else:
                concat_info[k].append(v)
    return concat_info

def merge_list_dicts(dict_of_lists1, dict_of_lists2):
    for k,v in dict_of_lists1.items():
        assert type(v) == list, "Error not a list."
        if k not in dict_of_lists2.keys():
            dict_of_lists2[k] = []
        dict_of_lists2[k] += v
    return dict_of_lists2
