# import os
import subprocess
import numpy as np
import random
import csv
from scipy.interpolate import interp1d
from stable_baselines3.common import logger

import sys
import inspect


def check_all_dict_values_equal(this_dict, until_idx=None):
    all_equal = True
    last_vals = None
    for k,v in this_dict.items():
        if until_idx is None:
            vals = v
        else:
            vals = v[:until_idx]
        if last_vals is None:
            last_vals = vals
            continue
        if vals != last_vals:
            all_equal = False
            break
        last_vals = vals
    return all_equal

# def interpolate_data(data, key_to_align):
#     align_val_lists = data[key_to_align].copy()
#     all_vals = []
#     avg_n_vals = 0
#     for run_id, val_list in align_val_lists.items():
#         all_vals += val_list
#         avg_n_vals += len(val_list)
#     avg_n_vals /= len(align_val_lists)
#     all_vals = sorted(list(set(all_vals)))
#     for col_key, col_data in data.items():
#         runs_to_del = []
#         for run_id, vals in col_data.items():
#             this_align_vals = align_val_lists[run_id]
#             # interpolate only if there are at least two values.
#             do_interp = True
#             if len(vals) < 2:
#                 do_interp = False
#             if do_interp:
#                 interp = interp1d(this_align_vals, vals, assume_sorted=True, fill_value='extrapolate')
#                 max_val_idx = all_vals.index(max(this_align_vals))
#                 # interpolate, but only up to max value of original data.
#                 vals_new = interp(all_vals[:max_val_idx+1])
#                 col_data[run_id] = vals_new
#             else:
#                 logger.info("Not considering run {} for plotting because too little data is available".format(run_id))
#                 runs_to_del.append(run_id)
#         for rtd in runs_to_del:
#             del col_data[rtd]
#     return data

def interpolate_data(col_data, align_val_lists):
    all_vals = []

    for run_id, val_list in align_val_lists.items():
        all_vals += list(val_list)


    all_vals = sorted(list(set(all_vals)))
    runs_to_del = []
    for run_id, vals in col_data.items():
        this_align_vals = align_val_lists[run_id]
        # interpolate only if there are at least two values.
        do_interp = True
        if len(vals) < 2:
            do_interp = False
        if do_interp:
            try:
                interp = interp1d(this_align_vals, vals, assume_sorted=True, fill_value='extrapolate')
                max_val_idx = all_vals.index(max(this_align_vals))
                # interpolate, but only up to max value of original data.
                vals_new = interp(all_vals[:max_val_idx+1])
                col_data[run_id] = vals_new
            except:
                pass
        else:
            logger.info("Not considering run {} for plotting because too little data is available".format(run_id))
            runs_to_del.append(run_id)
    for r in runs_to_del:
        del col_data[r]
    return col_data

def get_subdir_by_params(path_params, ctr=0):
    param_strs = []

    def shorten_split_elem(elem_str, chars_to_split):
        split_elems = elem_str.split(chars_to_split[0])
        short_split_elem_strs = []
        for split_elem in split_elems:
            if len(chars_to_split) == 1:
                if split_elem.find("_") == -1:
                    short_split_elem = str(split_elem)
                else:
                    short_split_elem = "_".join([us_elem[:2] for us_elem in split_elem.split("_")])
            else:
                short_split_elem = shorten_split_elem(split_elem, chars_to_split[1:])
            short_split_elem_strs.append(short_split_elem)
        short_ret_str = chars_to_split[0].join(short_split_elem_strs)
        return short_ret_str

    for p,v in sorted(path_params.items()):
        if str(v) == '':
            continue
        this_key_str = "".join([s[:3] for s in p.split("_")])
        chars_to_split = [",", ":", "[", "]"]
        this_v_str = shorten_split_elem(str(v), chars_to_split)
        this_param_str = '{}={}'.format(this_key_str, this_v_str)
        param_strs.append(this_param_str)

    subdir_str = "&".join(param_strs)
    subdir_str += "&" + str(ctr)

    # param_subdir = "_".join(
    #     ['{}:{}'.format("".join([s[:2] for s in p.split("_")]), str(v).split(":")[-1]) for p, v in
    #      sorted(path_params.items()) if str(v) != '']) + "_" + str(ctr)
    return subdir_str

def get_git_label():
    try:
        git_label = str(subprocess.check_output(["git", 'describe', '--always'])).strip()[2:-3]
    except:
        git_label = ''
    return git_label

def log_dict(params, logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))

def print_dict(params):
    ret_str = ''
    for key in sorted(params.keys()):
        ret_str+= '{}: {}\n'.format(key, params[key])
    return ret_str

def get_last_epoch_from_logdir(logdir):
    csv_file = logdir + '/progress.csv'
    max_epoch = 0
    try:
        with open(csv_file) as f:
            # data = f.read()
            epoch_idx = 0
            reader = csv.reader(f, delimiter=',', quotechar='|')
            for line, row in enumerate(reader):
                if line == 0:
                    for k_idx,k in enumerate(row):
                        if k == 'epoch':
                            epoch_idx = k_idx
                else:
                    max_epoch = max(int(row[epoch_idx]), max_epoch)
    except:
        return 0
    return max_epoch

def set_global_seeds(i):
    try:
        import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    except ImportError:
        rank = 0

    myseed = i  + 1000 * rank if i is not None else None
    try:
        import tensorflow as tf
        tf.random.set_seed(myseed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(myseed)
    except ImportError:
        pass
    np.random.seed(myseed)
    random.seed(myseed)




def get_full_size(obj, seen=None):
    """Recursively finds size of objects in bytes"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if hasattr(obj, '__dict__'):
        for cls in obj.__class__.__mro__:
            if '__dict__' in cls.__dict__:
                d = cls.__dict__['__dict__']
                if inspect.isgetsetdescriptor(d) or inspect.ismemberdescriptor(d):
                    size += get_full_size(obj.__dict__, seen)
                break
    if isinstance(obj, dict):
        size += sum((get_full_size(v, seen) for v in obj.values()))
        size += sum((get_full_size(k, seen) for k in obj.keys()))
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        for i in obj:
            try:
                size += get_full_size(i, seen)
            except Exception as e:
                print(f" could not determine size of {str(i)} because {e}")
        # size += sum((get_full_size(i, seen) for i in obj))

    if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
        size += sum(get_full_size(getattr(obj, s), seen) for s in obj.__slots__ if hasattr(obj, s))

    return size
# def import_function(spec):
#     """Import a function identified by a string like "pkg.module:fn_name".
#     """
#     mod_name, fn_name = spec.split(':')
#     module = importlib.import_module(mod_name)
#     fn = getattr(module, fn_name)
#     return fn
#
# def flatten_grads(var_list, grads):
#     """Flattens a variables and their gradients.
#     """
#     if len(var_list) == 0:
#         return []
#     try:
#         grad_list = [tf.reshape(grad, [U.numel(v)]) for (v, grad) in zip(var_list, grads)]
#     except Exception as e:
#         print(e)
#     grad_list = [tf.reshape(grad, [U.numel(v)]) for (v, grad) in zip(var_list, grads)]
#     return tf.concat(grad_list, 0)
#
# def flatten_grads_compact(var_list, grads):
#     """Flattens a variables and their gradients.
#     """
#     if len(var_list) == 0:
#         return []
#     return tf.concat([tf.reshape(grad, [U.numel(v)])
#                       for (v, grad) in zip(var_list, grads)], 0)
#
#
# def nn(input, layers_sizes, reuse=None, flatten=False, name=""):
#     """Creates a simple neural network
#     """
#     for i, size in enumerate(layers_sizes):
#         activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
#         input = tf.layers.dense(inputs=input,
#                                 units=size,
#                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
#                                 reuse=reuse,
#                                 name=name + '_' + str(i))
#         if activation:
#             input = activation(input)
#     if flatten:
#         assert layers_sizes[-1] == 1
#         input = tf.reshape(input, [-1])
#     return input
#
#
# def critic_nn(input, layers_sizes, q_limit, reuse=None, flatten=False, name=""):
#     ret_nn = nn(input, layers_sizes, reuse=reuse, flatten=flatten, name=name)
#     q_init = -0.067
#     # q_limit = q_limit
#     q_offset = -np.log(q_limit / q_init - 1)
#     output = tf.sigmoid(ret_nn + q_offset) * q_limit
#     return output
#
#
# def install_mpi_excepthook():
#     import sys
#     from mpi4py import MPI
#     old_hook = sys.excepthook
#
#     def new_hook(a, b, c):
#         old_hook(a, b, c)
#         sys.stdout.flush()
#         sys.stderr.flush()
#         MPI.COMM_WORLD.Abort()
#     sys.excepthook = new_hook
#
#
# def mpi_fork(n, extra_mpi_args=[]):
#     """Re-launches the current script with workers
#     Returns "parent" for original parent, "child" for MPI children
#     """
#     if n <= 1:
#         return "child"
#     if os.getenv("IN_MPI") is None:
#         env = os.environ.copy()
#         env.update(
#             MKL_NUM_THREADS="1",
#             OMP_NUM_THREADS="1",
#             IN_MPI="1"
#         )
#         # "-bind-to core" is crucial for good performance
#         args = ["mpirun", "-np", str(n)] + \
#             extra_mpi_args + \
#             [sys.executable]
#
#         args += sys.argv
#         ret = subprocess.check_call(args, env=env)
#         print(ret)
#         return "parent"
#     else:
#         install_mpi_excepthook()
#         return "child"
#
#
# def convert_episode_to_batch_major(episode):
#     """Converts an episode to have the batch dimension in the major (first)
#     dimension.
#     """
#     episode_batch = {}
#     for key in episode.keys():
#         val = np.array(episode[key]).copy()
#         # make inputs batch-major instead of time-major
#         episode_batch[key] = val.swapaxes(0, 1)
#
#     return episode_batch
#
#
# def transitions_in_episode_batch(episode_batch):
#     """Number of transitions in a given episode batch.
#     """
#     shape = episode_batch['u'].shape
#     return shape[0] * shape[1]
#
#
# def reshape_for_broadcasting(source, target):
#     """Reshapes a tensor (source) to have the correct shape and dtype of the target
#     before broadcasting it with MPI.
#     """
#     dim = len(target.get_shape())
#     shape = ([1] * (dim - 1)) + [-1]
#     return tf.reshape(tf.cast(source, target.dtype), shape)
#
# def prob_dist2discrete(prob_dist):
#     discrete = np.argmax(prob_dist, axis=-1)
#     # discrete = np.reshape(discrete, newshape=(prob_dist.shape[0],-1))
#     return discrete
#
#
# def physical_cpu_core_count():
#     try:
#         res = open('/proc/cpuinfo').read()
#         idx = res.find('cpu cores') + len("cpu cores")
#         idx = res.find(": ", idx) + len(": ")
#         nl_idx = res.find("\n", idx)
#         res = res[idx:nl_idx]
#         res = int(res)
#
#         if res > 0:
#             return res
#     except IOError:
#         return 0
#         pass
#
