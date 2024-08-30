import argparse
import itertools
import os
from typing import Any, Dict, List
from triton.testing import do_bench as triton_do_bench


# pylint: disable=unused-argument
def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, fast_flush=True, return_mode="mean",
             device="xpu", sync_submitting=True):
    assert return_mode in ["min", "max", "mean", "median"]
    import torch
    times = triton_do_bench(fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none, fast_flush=fast_flush,
                            return_mode="all", device_type=device)
    times = torch.tensor(times, dtype=torch.float)
    if quantiles is not None:
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if times.numel() > 2:
            # exclude max and min times
            times = torch.sort(times).values[1:-1]
        # add coefficient of the variance.
        std = torch.std(times)
        mean = torch.mean(times)
        cv = std / mean
        ret.extend([mean.tolist(), cv.tolist()])
        if len(ret) == 1:
            ret = ret[0]
        return ret
    return getattr(torch, return_mode)(times).item()


def assert_close(x, y, atol=None, rtol=None, err_msg=""):
    import numpy as np
    import torch

    # canonicalize arguments to be tensors
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y)
    # absolute tolerance
    if atol is None:
        atol = 1e-2
    atol = atol(x.dtype) if callable(atol) else atol
    # relative tolerance hook
    if rtol is None:
        rtol = 0.
    rtol = rtol(x.dtype) if callable(rtol) else rtol
    # we use numpy instead of pytorch
    # as it seems more memory efficient
    # pytorch tends to oom on large tensors
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        if y.dtype == torch.bfloat16:
            y = y.float()
        y = y.cpu().detach().numpy()
    # we handle size==1 case separately as we can
    # provide better error message there
    if x.size > 1 or y.size > 1:
        np.testing.assert_allclose(x, y, atol=atol, rtol=rtol, equal_nan=True)
        return
    if not np.allclose(x, y, atol=atol, rtol=rtol):
        raise AssertionError(f"{err_msg} {x} is not close to {y} (atol={atol}, rtol={rtol})")


def perf_report(benchmarks):
    """
    Mark a function for benchmarking. The benchmark can then be executed by using the :code:`.run` method on the return value.

    :param benchmarks: Benchmarking configurations.
    :type benchmarks: List of :class:`Benchmark`
    """
    wrapper = lambda fn: Mark(fn, benchmarks)
    return wrapper


# # pylint: disable=too-many-instance-attributes
class Benchmark:
    """
    This class is used by the :code:`perf_report` function to generate line plots with a concise API.
    """

    def __init__(
        self,
        x_names: List[str],
        x_vals: List[Any],
        line_arg: str,
        line_vals: List[Any],
        line_names: List[str],
        plot_name: str,
        args: Dict[str, Any],
        xlabel: str = "",
        ylabel: str = "",
        x_log: bool = False,
        y_log: bool = False,
        color=None,  # pylint: disable=unused-argument
        styles=None,
    ):
        """
        Constructor.
        x_vals can be a list of scalars or a list of tuples/lists. If x_vals is a list
        of scalars and there are multiple x_names, all arguments will have the same value.
        If x_vals is a list of tuples/lists, each element should have the same length as
        x_names.

        :param x_names: Name of the arguments that should appear on the x axis of the plot.
        :type x_names: List[str]
        :param x_vals: List of values to use for the arguments in :code:`x_names`.
        :type x_vals: List[Any]
        :param line_arg: Argument name for which different values correspond to different lines in the plot.
        :type line_arg: str
        :param line_vals: List of values to use for the arguments in :code:`line_arg`.
        :type line_vals: List[Any]
        :param line_names: Label names for the different lines.
        :type line_names: List[str]
        :param plot_name: Name of the plot.
        :type plot_name: str
        :param args: Dictionary of keyword arguments to remain fixed throughout the benchmark.
        :type args: Dict[str, Any]
        :param xlabel: Label for the x axis of the plot.
        :type xlabel: str, optional
        :param ylabel: Label for the y axis of the plot.
        :type ylabel: str, optional
        :param x_log: Whether the x axis should be log scale.
        :type x_log: bool, optional
        :param y_log: Whether the y axis should be log scale.
        :type y_log: bool, optional
        """
        self.x_names = x_names
        self.x_vals = x_vals
        self.x_log = x_log
        self.line_arg = line_arg
        self.line_vals = line_vals
        self.line_names = line_names
        self.y_log = y_log
        self.styles = styles
        # plot info
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.plot_name = plot_name
        self.args = args


class Mark:

    def __init__(self, fn, benchmarks):
        self.fn = fn
        self.benchmarks = benchmarks

    # pylint: disable=too-many-branches
    def _run(self, bench: Benchmark, save_path: str, show_plots: bool, print_data: bool, diff_col=False,
             save_precision=6, **kwrags):
        import matplotlib.pyplot as plt
        import pandas as pd
        y_vals = []
        for label in bench.ylabel:
            y_mean = [f"{x}-{label}" for x in bench.line_names]
            y_min = [f"{x}-{label}-min" for x in bench.line_names]
            y_max = [f"{x}-{label}-max" for x in bench.line_names]
            y_vals += y_mean + y_min + y_max
        y_vals += [f"{x}-CV" for x in bench.line_names]
        x_names = list(bench.x_names)
        df = pd.DataFrame(columns=x_names + y_vals)
        for x in bench.x_vals:
            # x can be a single value or a sequence of values.
            if not isinstance(x, (list, tuple)):
                x = [x for _ in x_names]

            if len(x) != len(x_names):
                raise ValueError(f"Expected {len(x_names)} values, got {x}")
            x_args = dict(zip(x_names, x))

            row_vals = {}
            for label in itertools.chain(bench.ylabel, ["CV"]):
                row_vals[label] = ([], [], [])
            for y in bench.line_vals:
                ret = self.fn(**x_args, **{bench.line_arg: y}, **bench.args, **kwrags)
                for i, label in enumerate(itertools.chain(bench.ylabel, ["CV"])):
                    try:
                        y_mean, y_min, y_max = ret[i]
                    except TypeError:
                        y_mean, y_min, y_max = ret[i], None, None
                    row_vals[label][0].append(y_mean)
                    row_vals[label][1].append(y_min)
                    row_vals[label][2].append(y_max)
            rows = []
            for label in bench.ylabel:
                if len(row_vals[label][0]) > 0:
                    rows += row_vals[label][0]
                if len(row_vals[label][1]) > 0:
                    rows += row_vals[label][1]
                if len(row_vals[label][2]) > 0:
                    rows += row_vals[label][2]
            rows += row_vals["CV"][0]
            df.loc[len(df)] = list(x) + rows

        if bench.plot_name:
            plt.figure()
            ax = plt.subplot()
            # Plot first x value on x axis if there are multiple.
            first_x = x_names[0]
            for label in bench.ylabel:
                for i, y in enumerate(bench.line_names):
                    y = f"{y}-{label}"
                    y_min, y_max = df[y + "-min"], df[y + "-max"]
                    col = bench.styles[i][0] if bench.styles else None
                    sty = bench.styles[i][1] if bench.styles else None
                    ax.plot(df[first_x], df[y], label=y, color=col, ls=sty)
                    if not y_min.isnull().all() and not y_max.isnull().all():
                        y_min = y_min.astype(float)
                        y_max = y_max.astype(float)
                        ax.fill_between(df[first_x], y_min, y_max, alpha=0.15, color=col)
            ax.legend()
            ax.set_xlabel(bench.xlabel or first_x)
            ax.set_ylabel(bench.ylabel)
            # ax.set_title(bench.plot_name)
            ax.set_xscale("log" if bench.x_log else "linear")
            ax.set_yscale("log" if bench.y_log else "linear")
            if show_plots:
                plt.show()
            if save_path:
                plt.savefig(os.path.join(save_path, f"{bench.plot_name}.png"))
        # df = df[x_names + bench.line_names]
        if diff_col and df.shape[1] == 2:
            col0, col1 = df.columns.tolist()
            df["Diff"] = df[col1] - df[col0]

        if print_data:
            print(bench.plot_name + ":")
            print(df.to_string())
        if save_path:
            df.to_csv(os.path.join(save_path, f"{bench.plot_name}.csv"), float_format=f"%.{save_precision}f",
                      index=False)
        return df

    def run(self, show_plots=False, print_data=False, save_path="", return_df=False, **kwargs):
        save_path = save_path_from_args(save_path)
        has_single_bench = isinstance(self.benchmarks, Benchmark)
        benchmarks = [self.benchmarks] if has_single_bench else self.benchmarks
        result_dfs = []

        for bench in benchmarks:
            result_dfs.append(self._run(bench, save_path, show_plots, print_data, **kwargs))

        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "results.html"), "w", encoding="utf-8") as html:
                html.write("<html><body>\n")
                for bench in benchmarks:
                    html.write(f"<image src=\"{bench.plot_name}.png\"/>\n")
                html.write("</body></html>\n")

        if return_df:
            return result_dfs[0] if has_single_bench else result_dfs

        return None


def save_path_from_args(save_path: str):
    """Returns a save path that is specified as an argument or via --reports comman line option."""
    if save_path:
        return save_path
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reports",
        type=str,
        default="",
        help="directory to save reports",
    )
    args = parser.parse_args()
    return args.reports