from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from functools import lru_cache
import numpy as np
import pandas as pd
import os

# Specifying what functions are accessible outside of the module
__all__ = [
    'index_to_alphabet',
    'directory_integrity_check',
    'default_fig_ax',
    'default_colors',
    'plot_graphs',
    'inverse_axis_conversion',
    'plt'
]


def index_to_alphabet(num):
    # Converting to base 26
    if num == 0:
        digits = [0]
    else:
        digits = []
        while num:
            digits.append(num % 26)
            num //= 26
        digits = digits[::-1]

    # Converting to characters
    digits[-1] += 1
    output = ""
    for digit in digits:
        output += chr(digit+96)
    return output


def directory_integrity_check(directory_name):
    """
    Makes sure a given directory exists
    """
    if not os.path.isdir(directory_name):
        os.mkdir(directory_name)
        print(f"Created {directory_name}")


@lru_cache(maxsize=1)
def default_fig_ax(fig_size):
    """
    Caches a version the default version of fig_size so it only has to be computed once
    (Only storing one fig_size's fig and ax)
    """
    return plt.subplots(figsize=fig_size)


@lru_cache(maxsize=1)
def default_colors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_graphs(xs, ys,
                x_label=None,
                y_label=None,
                labels=None,
                grid=True,
                x_log_scale=False,
                y_log_scale=False,
                show_plot=True,
                file_name=None,
                top_x_map=None,
                top_axis_label=None,
                fig_size=None,
                high_resolution=False,
                x_applied_function=None,
                y_applied_function=None,
                subplots=None,
                subplot_index=None,
                return_subplots=False,
                rasterized=True,
                show_legend=True,
                legend_loc="center left",
                legend_bbox_to_anchor=(1, 0.5),
                subplot_label=None,
                num_top_axis_bins=7,
                top_axis_prune='both',
                colors=default_colors()):

    if subplots is None:
        if fig_size is None:
            if labels is not None and show_legend:
                fig_size = (8.4, 4.8)
            else:
                fig_size = (6.4, 4.8)
        fig, ax = default_fig_ax(fig_size)
    else:
        fig, axes = subplots
        if subplot_index is not None:
            ax = axes[subplot_index]
        else:
            raise ValueError("subplot_index can't have value None when a custom subplot is used")

    if subplot_label is not None:
        ax.text(0.95, 0.95, subplot_label, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='left')

    if high_resolution:
        if show_plot:
            plt.rcParams["figure.dpi"] = 300
        if file_name:
            plt.rcParams["savefig.dpi"] = 300

    xs = list(xs)
    ys = list(ys)

    if x_applied_function is not None:
        xs = [x_applied_function(x.copy()) for x in xs]
    if y_applied_function is not None:
        ys = [y_applied_function(np.where(y > 0, y, np.nan)) for y in ys]

    if (labels is not None):
        labels = list(labels)
        i = -1
        for x, y, label in zip(xs, ys, labels):
            i += 1
            ax.plot(x, y, label=label, rasterized=rasterized, color=colors[i % len(colors)])
        if show_legend:
            ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)
    else:
        i = -1
        for x, y in zip(xs, ys):
            i += 1
            ax.plot(x, y, rasterized=rasterized, color=colors[i % len(colors)])

    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    if x_log_scale:
        ax.set_xscale('log')

    if y_log_scale:
        ax.set_yscale('log')

    if grid:
        ax.grid(True, which="both", ls="-", alpha=0.5)

    if top_x_map is not None:

        top_axis = ax.secondary_xaxis("top", functions=top_x_map)

        if top_axis_label is not None:
            top_axis.set_xlabel(top_axis_label, labelpad=10)

        # Ensure only finite ticks are used
        ticks = top_axis.get_xticks()

        # Gets rid of it trying to plot invalid log values.
        finite_ticks = ticks[np.isfinite(ticks)]
        top_axis.set_xticks(finite_ticks)

        top_axis.xaxis.set_major_locator(MaxNLocator(nbins=num_top_axis_bins, prune=top_axis_prune))

        """
        # Hide overlapping labels alternative (sort of worked)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        labels = top_axis.get_xticklabels()

        if len(labels) > 1:
            last_pos = labels[0].get_window_extent(renderer=renderer).x0

            for i in range(1, len(labels)-1):
                bbox = labels[i].get_window_extent(renderer=renderer)
                xpos = bbox.x0

                if (last_pos - xpos) < 0.8 * bbox.width:
                    labels[i].set_visible(False)
                else:
                    last_pos = xpos
        """

    plt.tight_layout()

    if show_plot:
        plt.show()

    if file_name:
        fig.savefig(file_name)

    if return_subplots:
        if subplot_index is not None:
            axes[subplot_index] = ax
            """
            if top_x_map is not None:
                return (fig, axes), top_axis
            """
            return (fig, axes)
        else:
            """
            if top_x_map is not None:
                return (fig, axes), top_axis
            """
            return (fig, ax)

    ax.clear()


def inverse_axis_conversion(axis, scalling_factor=1):
    # Prevents a division by zero warning
    with np.errstate(divide='ignore', invalid='ignore'):
        return scalling_factor/axis


if __name__ == "__main__":
    from .constant import eV, k_B  # , R
    from .units import m_sq_to_cm_sq

    class Hempelmann_Points_to_Data:
        """
        Calculates E_a and D_0 from a collection of points
        from a graph where you aren't given the exact values
        """
        def __init__(self, T_celsius, D):
            self.T = np.array(T_celsius)+273.15
            self.inv_T = 1/(self.T)
            self.ln_D = np.log(D)
            self.calc_D_0_and_E_a()

        def calc_D_0_and_E_a(self):
            slope, intercept = np.polyfit(self.inv_T, self.ln_D, 1)
            self.D_0 = np.exp(intercept)
            self.E_a = -slope * k_B / eV

    directory_integrity_check("graphs")

    graphs_to_plot = pd.DataFrame([
        {
            "name": "Diffusion Coefficient",
            "plot": True
        },
        {
            "name": "Retention",
            "plot": True
        }
    ])

    # Arrhenius Diffusion Constant D vs temperature T graph
    if graphs_to_plot[graphs_to_plot['name'] == 'Diffusion Coefficient']['plot'].values[0]:
        exponent = 3
        scalling_factor = 10**exponent

        # R. Hempelmann poitns to E_a and D_0
        Hempelmann_Ta_D = Hempelmann_Points_to_Data([425, -130], [2.44e-5, 1.63e-9])
        Hempelmann_Ta_T = Hempelmann_Points_to_Data([100, -115], [2.69e-6, 3e-9])

        """
        solubility_df = pd.DataFrame([
            {
                "name": "R_Frauenfelder",
                "label": "W-H (R. Frauenfelder)",
                "T": np.linspace(1100, 2400, 1000),
                "S_0": 0.29,  # cm^2/s
                "E_a": 1.04,
                "unit": "eV"
            }
        ])
        """

        diffusion_coefficient_df = pd.DataFrame([
            {
                "name": "R_Frauenfelder",
                "label": "W-H R. Frauenfelder",
                "T": np.linspace(1100, 2500, 1000),
                "D_0": 4.1e-3,  # cm^2/s
                "E_a": 0.391
            },
            {
                "name": "R_Frauenfelder_2",
                "label": "W-H R. Frauenfelder 2",
                "T": np.linspace(1500, 2500, 1000),
                "D_0": 1.58e-3,  # cm^2/s
                "E_a": 0.25
            },
            {
                "name": "K_Heinola",
                "label": "W-H K. Heinola",
                "T": np.linspace(1500, 2500, 1000),
                "D_0": 5.2e-4,  # cm^2/s
                "E_a": 0.21
            },
            {
                "name": "D_Johnson",
                "label": "W-H D. Johnson",
                "T": np.linspace(1500, 2500, 1000),
                "D_0": 8.93e-3,  # cm^2/s
                "E_a": 0.38
            },
            {
                "name": "N_Degtyarenko",
                "label": "W-H N. Degtyarenko",
                "T": np.linspace(1400, 2000, 1000),
                "D_0": 2.3e-3,
                "E_a": 0.29
            },
            {
                "name": "G_Holzner",
                "label": "W-H G. Holzner",
                "T": np.linspace(1600, 2600, 1000),
                "D_0": 2.06e-3,
                "E_a": 0.28  # ± 0.06 eV
            },
            {
                "name": "G_Holzner",
                "label": "W-D G. Holzner",
                "T": np.linspace(1600, 2600, 1000),
                "D_0": 1.6e-3,
                "E_a": 0.28  # ± 0.06 eV
            },
            {
                "name": "J_Dark",
                "label": "W-D J. Dark",
                "T": np.linspace(370, 800, 1000),
                "D_0": m_sq_to_cm_sq(1.6e-7),
                "E_a": 0.28
            },
            {  # Gave extra values for Frauenfelder, Heinola, and Johnson
                "name": "M_Qiu",
                "label": "W-H M. Qiu",
                "T": np.linspace(300, 2000, 1000),
                "D_0": 8.03e-4,  # ± 0.012e-4 cm^2/s
                "E_a": 0.218  # ± 0.002 eV
            },
            {
                "name": "M_Qiu",
                "label": "W-D M. Qiu",
                "T": np.linspace(300, 2000, 1000),
                "D_0": 7.551e-4,  # ± 0.025e-4 cm^2/s
                "E_a": 0.225  # ± 0.001 eV
            },
            {
                "name": "M_Qiu",
                "label": "W-T M. Qiu",
                "T": np.linspace(300, 2000, 1000),
                "D_0": 6.902e-4,  # ± 0.012e-4 cm^2/s
                "E_a": 0.227  # ± 0.001 eV
            },
            {
                "name": "M_S_Anand",
                "label": "Ta-H M.S. Anand",
                "T": np.linspace(298, 373, 1000),
                "D_0": 4.38e-4,  # cm^2/s
                "E_a": 0.142  # eV
            },
            {
                "name": "G_Schaumann_H",
                "label": "Ta-H G. Schaumann",
                "T": np.linspace(223.15, 573.15, 1000),
                "D_0": 4.4e-4,  # ± 0.4e-4 cm^2/s
                "E_a": 0.14  # ± 0.004 eV
            },
            {
                "name": "G_Schaumann_D",
                "label": "Ta-D G. Schaumann",
                "T": np.linspace(273.15, 473.15, 1000),
                "D_0": 4.9e-4,  # ± 0.9e-4 cm^2/s
                "E_a": 0.163  # ± 0.006 eV
            },
            {
                "name": "R_Hempelmann",
                "label": "Ta-D R. Hempelmann",
                "T": np.linspace(Hempelmann_Ta_D.T.min(), Hempelmann_Ta_D.T.max(), 1000),
                "D_0": Hempelmann_Ta_D.D_0,
                "E_a": Hempelmann_Ta_D.E_a
            },
            {
                "name": "R_Hempelmann",
                "label": "Ta-T R. Hempelmann",
                "T": np.linspace(Hempelmann_Ta_T.T.min(), Hempelmann_Ta_T.T.max(), 1000),
                "D_0": Hempelmann_Ta_T.D_0,
                "E_a": Hempelmann_Ta_T.E_a
            }
        ])

        """
            {
                "name": "R_Frauenfelder_old",
                "label": "W-H R. Frauenfelder old",  # new one stated by G. Holzner
                "T": np.linspace(1100, 2400, 1000),
                "D_0": 4.1e-3,  # cm^2/s
                "E_a": 9000,  # J/mol
                "unit": "J/mol"
            },
            {
                "name": "M_Weiser",
                "label": "Ta-H M. Weiser",
                "T": np.linspace(150, 290, 1000),
                "D_0": m_sq_to_cm_sq(1e-18),
                "E_a": 0.270,  # ± 0.06 eV
                "unit": "eV"
            },
            {
                "name": "E_Moore",
                "label": "W-H E. Moore",
                "T": np.linspace(1200, 2500, 1000),
                "D_0": 7.25e-4,
                "E_a": 1.8,
            },
            {
                "name": "L_Ryabchikov",
                "label": "W-H L. Ryabchikov",
                "T": np.linspace(1900, 2400, 1000),
                "D_0": 8.1e-2,
                "E_a": 0.86,
            },
        """

        def compute_D(row, T=None):
            if T is None:
                T = row['T']

            return row['D_0'] * np.exp(-row['E_a'] / (k_B * T / eV))
            """
            if row['unit'] == "J/mol":
                return row['D_0'] * np.exp(-row['E_a'] / (R * T))
            elif row['unit'] == "eV":
                return row['D_0'] * np.exp(-row['E_a'] / (k_B * T / eV))
            """

        # Calculates D for each row (axis=1), rather than for each column axis=0)
        diffusion_coefficient_df['D'] = diffusion_coefficient_df.apply(compute_D, axis=1)

        """
        diffusion_coefficient_df.to_csv('diffusion_coefficients.csv', index=False)

        print(diffusion_coefficient_df.to_latex(index=False,
                                                formatters={"T": lambda T: fr"{T[0]:.0f}$\rightarrow${T[-1]:.0f}"},
                                                float_format="{:.2e}".format,
                                                columns=["label", "T", "D_0", "E_a"],
                                                label="table:diffusion_coefficients_table",
                                                escape=False,
                                                column_format="c c c c",
                                                position="H"))
        """

        def temp_to_inv_temp(temp):
            return inverse_axis_conversion(temp, scalling_factor)

        inv_temp_to_temp = temp_to_inv_temp

        print("Plotting diffusion coefficient graphs...")
        for element in ("W_and_Ta", "W", "Ta"):
            match element:
                case "W_and_Ta":
                    df_used = diffusion_coefficient_df
                    num_columns = 4
                case "W":
                    df_used = diffusion_coefficient_df[diffusion_coefficient_df['label'].str.startswith('W')]
                    num_columns = 3
                case "Ta":
                    df_used = diffusion_coefficient_df[diffusion_coefficient_df['label'].str.startswith('Ta')]
                    num_columns = 2
                case _:
                    raise ValueError("Invalid value for element")
            print(element.replace("_", " "))

            num_temps = len(df_used['T'])
            num_extra_graphs = 1
            num_rows = int(np.ceil((num_temps + num_extra_graphs)/num_columns))
            fig, axes = plt.subplots(num_rows, num_columns, figsize=(6*num_columns+2, 4*num_rows))
            subplots = (fig, axes)

            # Original Tempearture ranges
            subplots = plot_graphs(df_used['T'], df_used['D'],
                                   x_label=f"10$^{exponent}$/Temperature (Original)" + " [K$^{-1}$]",
                                   y_label="log(Diffusion Constant [cm$^2$/s])",
                                   labels=df_used['label'],
                                   x_applied_function=temp_to_inv_temp,
                                   y_applied_function=np.log10,
                                   top_x_map=(inv_temp_to_temp, temp_to_inv_temp),
                                   top_axis_label="Temperature [K]",
                                   show_plot=False,
                                   return_subplots=True,
                                   subplots=subplots,
                                   subplot_index=(0, 0),
                                   show_legend=False,
                                   subplot_label="(a)")

            # Each temperature range applied to every formula
            for i, T in enumerate(df_used['T']):
                subplot_index = divmod(i + num_extra_graphs, num_columns)
                Ts = [T.copy() for _ in range(num_temps)]
                Ds = df_used.apply(compute_D, axis=1, T=T)

                """
                plot_graphs(Ts, Ds,
                            x_label="Temperature [K]",
                            y_label="Diffusion Constant [cm$^2$/s]",
                            labels=df_used['label'],
                            file_name=(f"graphs/{element}_Diffusion_Constant" +
                                       f"_vs_{df_used['name'].iloc[i]}_temperatures.png"),
                            show_plot=False)
                """
                # Plots the subplot with a legend if there are no blank subplots
                if ((num_temps + num_extra_graphs) % num_columns == 0) and (subplot_index == (0, num_columns - 1)):
                    subplots = plot_graphs(Ts, Ds,
                                           x_label=f"10$^{exponent}$/Temperature ({df_used['label'].iloc[i]})" + " [K$^{-1}$]",
                                           y_label="ln(Diffusion Constant [cm$^2$/s])",
                                           labels=df_used['label'],
                                           x_applied_function=temp_to_inv_temp,
                                           y_applied_function=np.log10,
                                           top_x_map=(inv_temp_to_temp, temp_to_inv_temp),
                                           top_axis_label="Temperature [K]",
                                           show_plot=False,
                                           return_subplots=True,
                                           subplots=subplots,
                                           subplot_index=subplot_index,
                                           subplot_label=f"({index_to_alphabet(i + num_extra_graphs)})")
                # Once all the other subplots have been plotted:
                elif i == (num_temps - 1):
                    # Adds a legend to the graph if there were blank subplots
                    if (num_temps + num_extra_graphs) % num_columns != 0:
                        for j in range((num_temps + num_extra_graphs), axes.size):
                            # Make said blank suplots are invisible
                            axes[num_rows - 1, j % num_columns].axis("off")
                            if j == axes.size - 1:
                                handles, labels = axes[0, 1].get_legend_handles_labels()
                                axes[num_rows-1, -1].legend(handles, labels, loc="center", bbox_to_anchor=(0.5, 0.5))

                    plot_graphs(Ts, Ds,
                                x_label=f"10$^{exponent}$/Temperature ({df_used['label'].iloc[i]})" + " [K$^{-1}$]",
                                y_label="ln(Diffusion Constant [cm$^2$/s])",
                                labels=df_used['label'],
                                file_name=(f"graphs/{element}_Arrhenius_Diffusion_Constant.png"),
                                x_applied_function=temp_to_inv_temp,
                                y_applied_function=np.log10,
                                top_x_map=(inv_temp_to_temp, temp_to_inv_temp),
                                top_axis_label="Temperature [K]",
                                show_plot=False,
                                subplots=subplots,
                                subplot_index=subplot_index,
                                show_legend=False,
                                subplot_label=f"({index_to_alphabet(i + num_extra_graphs)})")
                else:
                    subplots = plot_graphs(Ts, Ds,
                                           x_label=f"10$^{exponent}$/Temperature ({df_used['label'].iloc[i]})" + " [K$^{-1}$]",
                                           y_label="ln(Diffusion Constant [cm$^2$/s])",
                                           labels=df_used['label'],
                                           x_applied_function=temp_to_inv_temp,
                                           y_applied_function=np.log10,
                                           top_x_map=(inv_temp_to_temp, temp_to_inv_temp),
                                           top_axis_label="Temperature [K]",
                                           show_plot=False,
                                           return_subplots=True,
                                           subplots=subplots,
                                           subplot_index=subplot_index,
                                           show_legend=False,
                                           subplot_label=f"({index_to_alphabet(i + num_extra_graphs)})")

    # Retention vs temperature
    if graphs_to_plot[graphs_to_plot['name'] == 'Retention']['plot'].values[0]:
        print("Plotting hydrogen isotope retention graph...")
        plt.clf()

        file_name = "hydrogen_isotope_retention_vs_temperature"
        subplots = plt.subplots(3, figsize=(6, 12))

        r_causey_retention = np.loadtxt(f"csvs/{file_name}_r_causey.csv", delimiter=",", dtype=float)
        # Sorting by temperature as digital graph to csv converter outputted the datapoints in a random order
        r_causey_retention = r_causey_retention[r_causey_retention[:, 0].argsort()]

        # This data is already sorted
        y_ueda_retention = np.loadtxt(f"csvs/{file_name}_y_ueda.csv", delimiter=",", dtype=float)

        colors = default_colors()

        subplots = plot_graphs([r_causey_retention[:, 0], y_ueda_retention[:, 0]],
                               [r_causey_retention[:, 1], y_ueda_retention[:, 1]],
                               x_label="Temperature (K)",
                               y_label="Hydrogen Isotope Rentention (10$^{20}$ isotopes/m$^2$)",
                               show_plot=False,
                               labels=["W-(D+T) R. Causey", "W-(D) Y. Ueda"],
                               subplots=subplots,
                               subplot_index=0,
                               return_subplots=True,
                               subplot_label="(a)",
                               legend_loc="best")

        subplots = plot_graphs([r_causey_retention[:, 0]], [r_causey_retention[:, 1]],
                               x_label="Temperature (K)",
                               y_label="Hydrogen Isotope Rentention (10$^{20}$ (D+T)/m$^2$)",
                               show_plot=False,
                               subplots=subplots,
                               subplot_index=1,
                               return_subplots=True,
                               subplot_label="(b)",
                               colors=[colors[0]])

        plot_graphs([y_ueda_retention[:, 0]], [y_ueda_retention[:, 1]],
                    x_label="Temperature (K)",
                    y_label="Deuterium Rentention (10$^{20}$ D/m$^2$)",
                    file_name=f"graphs/{file_name}.png",
                    show_plot=False,
                    subplots=subplots,
                    subplot_index=2,
                    subplot_label="(c)",
                    colors=[colors[1]])
    print("Done")

"""
    # H. TEICHLER and A. KLAMT
    # Haven't quite got working yet

    a_0 = 3.3029e-10
    k_B = 1.380649e-23
    beta = 1 / (k_B * T)
    eV = 1.6e-19
    I = np.array([0.22, 9.9]) * 1e-3 * eV
    E = np.array([20.6, 12.1]) * 1e-3 * eV
    h = 6.62607015e-34
    h_bar = h/(2*np.pi)
    omega = E/h_bar
    inv_tau = omega[0]/(2*np.pi)*np.exp(-E[0]*beta) + omega[1]/(2*np.pi)*np.exp(-E[1]*beta)
    print(inv_tau)
    epsilon_0 = 8.854e-12 # Not the epsilon it's talking about
    D = 1/12 * a_0 * a_0 * inv_tau * np.exp(-epsilon_0 * beta)/np.exp(-epsilon_0 * beta)
    print(D)

    plot(scalling_factor/T, np.log10(D), x_label=f"10$^{exponent}$/Temperature" + " [K$^{-1}$]",
         y_label="ln(Diffusion Constant [cm$^2$/s])", label="Hydrogen (H. TEICHLER and A. KLAMT)",
         top_x_map=(lambda x: inverse_axis_conversion(x, scalling_factor),
                   lambda x: inverse_axis_conversion(x, scalling_factor)),
                   top_axis_label="Temperature [K]")

    """
