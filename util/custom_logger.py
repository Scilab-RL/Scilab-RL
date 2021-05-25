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
from stable_baselines3.common.logger import Video, FormatUnsupportedError, SeqWriter
import warnings
import time
from cycler import cycler
import mlflow
import hydra
import omegaconf
from omegaconf import DictConfig, ListConfig


class MLFlowOutputFormat(KVWriter, SeqWriter):

    def __init__(self):
        self.log_txt = ''
        return

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:
        kvs = key_values.copy()
        for k,v in kvs.items():
            mlflow.log_metric(k, v, step=step)

    def write_sequence(self, sequence: List) -> None:
        sequence = list(sequence)
        for i, elem in enumerate(sequence):
            self.log_txt += elem + '\n'
        mlflow.log_text(self.log_txt, f"log.txt")

    def close(self) -> None:
        return

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


# def mlflow_log_text(self, run_id, text, artifact_file):
#     with self._log_artifact_helper(run_id, artifact_file) as tmp_path:
#         with open(tmp_path, "a") as f: ## Here is the change: map "w" to "a"
#             f.write(text)