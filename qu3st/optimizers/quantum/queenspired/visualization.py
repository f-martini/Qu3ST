import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display_functions import display, clear_output


def clear_n_plot(ax_name, recorder):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if ax_name in kwargs.keys():
                # clear ax
                kwargs[ax_name].cla()
                result = func(*args, **kwargs)
            else:
                raise KeyError("Missing ax to update.")
            # show the plot
            fig = kwargs[ax_name].get_figure()
            clear_output(wait=True)
            display(fig)
            # add frame to list of frames
            if recorder in kwargs.keys():
                # clear ax
                if kwargs[recorder] is not None:
                    fig.canvas.draw()
                    width, height = fig.get_size_inches() * fig.get_dpi()
                    image_np = np.frombuffer(fig.canvas.tostring_rgb(),
                                             dtype=np.uint8).reshape(
                        int(height), int(width), 3)
                    kwargs[recorder].add_frame(image_np)

        return wrapper

    return decorator


def _add_reference_lines(ax, energy_hard_cap=10 ** 7):
    ZERO = 0
    MIN_ENERGY = -1
    ax.axhline(y=energy_hard_cap, color='firebrick', linestyle='--',
               label='Energy Hard-cap')
    ax.axhline(y=ZERO, color='darkorange', linestyle='--',
               label=f'Reference Line {ZERO}')
    ax.axhline(y=MIN_ENERGY, color='forestgreen', linestyle='--',
               label=f'Reference Line {MIN_ENERGY}')


def _set_ax_properties(ax,
                       title,
                       xlabel,
                       ylabel,
                       ylim=(-2, 10 ** 8),
                       yscale="symlog",
                       legend_loc='center left',
                       linthresh=1):
    ax.set_ylim(ylim[0], ylim[1])
    if yscale == "symlog":
        ax.set_yscale(value=yscale, linthresh=linthresh)
    ax.set_ylabel(ylabel=ylabel)
    ax.set_xlabel(xlabel=xlabel)
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    return


@clear_n_plot(ax_name="ax", recorder="recorder")
def update_plot_loss(energy_vals, ax, title="", recorder=None):
    # prepare plot
    ax.plot([i for i in range(len(energy_vals))],
            [e.real for e in energy_vals],
            label="Energy Loss",
            color="cornflowerblue", )
    _add_reference_lines(ax, energy_hard_cap=2)
    _set_ax_properties(ax=ax,
                       title=title,
                       xlabel="Iterations",
                       ylabel="Energy",
                       ylim=(-1.1, 2),  # max(energy_vals)),
                       yscale="log"
                       )


@clear_n_plot(ax_name="ax", recorder="recorder")
def update_plot_sampling(energy_vals, energy_best,
                         ax, title="", xlabel="", recorder=None):
    ax.plot([i for i in range(len(energy_vals))],
            [min(10 ** 7, e.real) for e in energy_best],
            label="Best Energy Found",
            color="violet",
            marker='o')
    ax.plot([i for i in range(len(energy_vals))],
            [min(10 ** 7, e.real) for e in energy_vals],
            label="Energy Evaluations",
            color="cornflowerblue",
            marker='o')
    _add_reference_lines(ax)
    _set_ax_properties(ax=ax,
                       title=title,
                       xlabel=xlabel,
                       ylabel="Energy")
