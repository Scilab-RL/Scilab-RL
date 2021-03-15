import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from util.custom_logger import MatplotlibOutputFormat
import argparse

# help flag provides flag help
# store_true actions stores argument as True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', type=str, default='data', help="the logdir path to plot from")
    parser.add_argument('--cols_to_plot', type=str, default='test/success_rate,test/mean_reward', help="the data columns to plot.")
    args = parser.parse_args()
    cols = args.cols_to_plot.split(',')
    plotter = MatplotlibOutputFormat(args.logdir, 0, cols_to_plot=cols, plot_parent_dir=False)
    plotter.plot_aggregate_kvs()