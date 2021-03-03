import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import os
from stable_baselines3.common import logger
from stable_baselines3.common.logger import KVWriter
import matplotlib.pyplot as plt
from collections import OrderedDict
import csv
import numpy as np
from util.util import print_dict, check_all_dict_values_equal, interpolate_data
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union
from stable_baselines3.common.logger import Video, FormatUnsupportedError
import warnings

class MatplotlibOutputFormat(KVWriter):
    def __init__(self, logpath, cols_to_plot=['test/success_rate', 'test/mean_reward']):
        self.logpath = logpath
        self.csv_filename = "plot.csv"
        self.csv_filepath = logpath + "/plot.csv"
        self.file = open(self.csv_filepath, 'w+t')
        self.keys = []
        self.sep = ','
        self.data_read_dir = "/".join(logpath.split("/")[:-1])
        self.step = 1
        self.cols_to_plot = cols_to_plot
        self.plot_colors = sorted(plt.rcParams['axes.prop_cycle'].by_key()['color'], reverse=True)

    def write(self, kvs, name_to_excluded, step):
        # Add our current row to the history
        extra_keys = kvs.keys() - self.keys
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(',')
                self.file.write(k)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(',')
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write('\n')
        self.file.flush()
        self.plot_aggregate_kvs()

    def close(self):
        self.file.close()

    def plot_aggregate_kvs(self):
        def config_from_folder(folder_str):
            return "&".join(folder_str.split("&")[:-1])

        configs = set()
        config_data = {}
        data_dict = OrderedDict()
        config_str = ''
        for configdir in os.listdir(self.data_read_dir):
            if not os.path.isdir(os.path.join(self.data_read_dir,configdir)) or '.ipynb_checkpoints' in configdir:
                continue
            folder = self.data_read_dir+"/"+configdir
            config_str = config_from_folder(configdir)
            if config_str not in data_dict.keys():
                data_dict[config_str] = OrderedDict()
            config_ctr_str = configdir.split("&")[-1]
            configs.add(config_str)
            if config_str not in config_data.keys():
                config_data[config_str] = []
            try:
                with open(self.data_read_dir+"/"+configdir+"/"+self.csv_filename) as csvfile:
                    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                    for line, row in enumerate(reader):
                        if line == 0:
                            keys = row.copy()
                            for k in row:
                                if k not in data_dict[config_str].keys():
                                    data_dict[config_str][k] = {}
                                if config_ctr_str not in data_dict[config_str][k].keys():
                                    data_dict[config_str][k][config_ctr_str] = []
                        else:
                            for idx,item in enumerate(row):
                                #     print('huh')
                                try:
                                    data_dict[config_str][keys[idx]][config_ctr_str].append(float(item))
                                except:
                                    data_dict[config_str][keys[idx]][config_ctr_str].append(np.nan)
                                    pass
                                    # print("item is not a float")
            except Exception as e:
                logger.warn("Warning, could not plot data from {} because the file was not found.".format(self.data_read_dir+"/"+configdir+"/"+self.csv_filename))
        self.plot_dict(data_dict)

    def tolerant_median(self, data_dict):
        keys = sorted(data_dict.keys())
        arr_lens = []
        for k in keys:
            arr_lens.append(len(data_dict[k]))
        shortest_key_data_count = min(arr_lens)
        longest_key_data_count = max(arr_lens)
        shortest_key = ''
        longest_key = ''
        for k in keys:
            if len(data_dict[k]) == shortest_key_data_count:
                shortest_key = k
            if len(data_dict[k]) == longest_key_data_count:
                longest_key = k

        data_info = {'shortest_key': shortest_key, 'shortest_data_count': shortest_key_data_count,
                     'longest_key': longest_key, 'longest_dat_count': longest_key_data_count}

        d_lists = []
        for ctr in range(longest_key_data_count):
            d_lists.append([])
            for k in keys:
                if len(data_dict[k]) > ctr:
                    d_lists[-1].append(data_dict[k][ctr])

        median = [np.median(dlist) for dlist in d_lists]
        lower = [np.percentile(dlist, 25) for dlist in d_lists]
        upper = [np.percentile(dlist, 75) for dlist in d_lists]

        return median, upper, lower, data_info

    def plot_dict(self, data_dict):
        # First interpolate all data
        cols_to_del = []
        for config_str in data_dict.keys():
            data = data_dict[config_str]
            timesteps = data['time/total timesteps'].copy()
            for k in data.keys():
                if k in self.cols_to_plot + ['time/total timesteps']:
                    data_dict[config_str][k] = interpolate_data(data[k], timesteps)
                else:
                    cols_to_del.append(k)
            for k in cols_to_del:
                if k in data.keys():
                    del data_dict[config_str][k]

        for k in self.cols_to_plot:
            fig = plt.figure(figsize=(20, 10))
            color_idx = 0
            all_data_info = {}
            for config_str in data_dict.keys():
                data = data_dict[config_str]
                if k not in data.keys():
                    continue
                median, upper, lower, data_info = self.tolerant_median(data[k])
                min_data_len = data_info['shortest_data_count']
                if 'time/total timesteps' in data.keys():
                    xs = data['time/total timesteps'][data_info['shortest_key']]
                    xs_label = 'action steps'
                else:
                    xs = range(0, data_info['shortest_data_count'])
                    xs_label = 'epochs'
                all_data_info[config_str] = data_info
                plt.plot(xs, median[:min_data_len], color=self.plot_colors[color_idx], label=config_str + '-' + k)
                plt.fill_between(xs, lower[:min_data_len],
                                 upper[:min_data_len],
                                 alpha=0.25, color=self.plot_colors[color_idx])
                plt.xlabel(xs_label)
                plt.ylabel(k)
                color_idx += 1
                if color_idx >= len(self.plot_colors):
                    color_idx = 0
            key_str = k.replace('/','-')
            plot_log_filename = self.data_read_dir + '/' + key_str + '.log'
            plot_filename = self.data_read_dir+'/'+key_str+'.png'
            plt.legend()
            plt.savefig(plot_filename, format='png')
            plt.close(fig)
            with open(plot_log_filename, 'w') as logfile:
                logfile.write(print_dict(all_data_info))


class SeqWriter(object):
    """
    sequence writer
    """

    def write_sequence(self, sequence: List) -> None:
        """
        write_sequence an array to file

        :param sequence:
        """
        raise NotImplementedError


class FixedHumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file: Union[str, TextIO]):
        """
        log to a file, in a human readable format

        :param filename_or_file: the file to write the log to
        """
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "write"), f"Expected file or str, got {filename_or_file}"
            self.file = filename_or_file
            self.own_file = False

    def write(self, key_values: Dict, key_excluded: Dict, step: int = 0) -> None:
        # Create strings for printing
        kv_list = {}
        # key2str = {}
        tag = None
        for (key, value), (_, excluded) in zip(sorted(key_values.items()), sorted(key_excluded.items())):

            if excluded is not None and ("stdout" in excluded or "log" in excluded):
                continue

            if isinstance(value, Video):
                raise FormatUnsupportedError(["stdout", "log"], "video")

            if isinstance(value, float):
                # Align left
                value_str = f"{value:<8.3g}"
            else:
                value_str = str(value)

            if key.find("/") > 0:  # Find tag and add it to the dict
                tag = key[: key.find("/") + 1]
            else:
                tag = ''
            if tag not in kv_list.keys():
                # key2str[self._truncate(tag)] = ""
                kv_list[self._truncate(tag)] = {}

            key = str("   " + key[len(tag):])
            val = self._truncate(value_str)
            kv_list[tag][key] = val

        # Find max widths
        if len(kv_list.keys()) == 0:
            warnings.warn("Tried to write empty key-value dict")
            return
        else:
            key_width = 0
            val_width = 0
            for tag, vlist in kv_list.items():
                for k,v in vlist.items():
                    key_width = max(key_width, len(k))
                    val_width = max(val_width, len(v))

        # Write out the data
        dashes = "-" * (key_width + val_width + 7)
        lines = [dashes]
        for tag, vlist in kv_list.items():
            key_space = " " * (key_width - len(tag))
            val_space = " " * (val_width)
            lines.append(f"| {tag}{key_space} | {val_space} |")
            for k,v in vlist.items():
                val_space = " " * (val_width - len(v))
                key_space = " " * (key_width - len(k))
                lines.append(f"| {k}{key_space} | {v}{val_space} |")
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")
        self.file.flush()

    @classmethod
    def _truncate(cls, string: str, max_length: int = 33) -> str:
        return string[: max_length - 3] + "..." if len(string) > max_length else string

    def write_sequence(self, sequence: List) -> None:
        sequence = list(sequence)
        for i, elem in enumerate(sequence):
            self.file.write(elem)
            if i < len(sequence) - 1:  # add space unless this is the last one
                self.file.write(" ")
        self.file.write("\n")
        self.file.flush()

    def close(self) -> None:
        """
        closes the file
        """
        if self.own_file:
            self.file.close()