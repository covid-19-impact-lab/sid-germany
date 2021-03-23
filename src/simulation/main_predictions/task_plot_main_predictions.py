import dask.dataframe as dd
import pandas as pd
import pytask
import seaborn as sns

from src.config import BLD
from src.config import SRC
from src.simulation.main_specification import build_main_scenarios
from src.simulation.main_specification import PREDICT_PATH
from src.simulation.main_specification import SCENARIO_START
from src.simulation.plotting import calculate_virus_strain_shares
from src.simulation.plotting import plot_incidences
from src.simulation.plotting import weekly_incidences_from_results

NESTED_PARAMETRIZATION = build_main_scenarios(PREDICT_PATH)

DEPENDENCIES = {
    "specs_py": SRC / "simulation" / "main_specification.py",
    "plotting_py": SRC / "simulation" / "plotting.py",
}
for name, seed_list in NESTED_PARAMETRIZATION.items():
    for path, _, seed in seed_list:
        DEPENDENCIES[(name, seed)] = path

INCIDENCE_PATHS = {
    "all_incidences": PREDICT_PATH / "all_incidences.pkl",
    "mean_incidences": PREDICT_PATH / "mean_incidences.csv",
    "all_strain_shares": PREDICT_PATH / "all_strain_shares.pkl",
    "mean_strain_shares": PREDICT_PATH / "mean_strain_shares.csv",
}

NAME_TO_LABEL = {
    "base_scenario": "Status Quo beibehalten",
    "november_home_office_level": "Home-Office-Quote wie im November",
    "spring_home_office_level": "Home-Office-Quote wie im 1. Lockdown",
    "emergency_child_care": "Bildungseinrichtungen bieten nur Notbetreuung an",
}


@pytask.mark.depends_on(DEPENDENCIES)
@pytask.mark.produces(INCIDENCE_PATHS)
def task_save_statistics_for_main_scenario(depends_on, produces):
    # specs is not directly used as input, so drop it
    depends_on.pop("specs_py")
    depends_on.pop("plotting_py")

    results = {}
    scenario_names = NESTED_PARAMETRIZATION.keys()
    for name in scenario_names:
        paths = [path for (scenario, _), path in depends_on.items() if name in scenario]
        results[name] = [dd.read_parquet(p) for p in paths]

    incidences, mean_incidences = _get_incidences_from_results(results)
    pd.to_pickle(incidences, produces["all_incidences"])
    mean_incidences.to_csv(produces["mean_incidences"])

    strain_shares = {
        name: calculate_virus_strain_shares(runs) for name, runs in results.items()
    }
    pd.to_pickle(strain_shares, produces["all_strain_shares"])

    to_concat = [strain_shares[name].mean(axis=1, level=0) for name in scenario_names]
    mean_shares = pd.concat(to_concat, keys=scenario_names, names=["scenario"])
    mean_shares.to_csv(produces["mean_strain_shares"])


def _get_incidences_from_results(results):
    """Create a dictionary with all incidences over time and the mean over runs.

    Args:
        results (dict):

    Returns:
        incidences (dict): Keys are (scenario_name, outcome) tuples.
            Values are DataFrames with one column per run.
        mean_incidences (pd.DataFrame): one column per scenario and outcome with
            the mean incidence of that scenario.

    """
    incidences = {}
    for outcome in ["new_known_case", "newly_infected"]:
        for name, simulation_runs in results.items():
            time_series_df = weekly_incidences_from_results(simulation_runs, outcome)
            incidences[(name, outcome)] = time_series_df

    to_concat = [
        pd.Series(df.mean(axis=1), name=name) for name, df in incidences.items()
    ]
    mean_incidences = pd.concat(to_concat, axis=1)
    return incidences, mean_incidences


PLOT_DEPENDENCIES = {
    "plotting_py": SRC / "simulation" / "plotting.py",
    "rki_data": BLD / "data" / "processed_time_series" / "rki.pkl",
    "rki_strains": BLD / "data" / "virus_strains" / "rki_strains.csv",
    **INCIDENCE_PATHS,
}

PLOT_PARAMETRIZATION = []
for outcome, title in [
    ("new_known_case", "Beobachtete Inzidenz"),
    ("newly_infected", "Tatsächliche Inzidenz"),
]:
    produces = {"fig": PREDICT_PATH / f"{outcome}.png"}
    spec = (outcome, title, produces)
    PLOT_PARAMETRIZATION.append(spec)


@pytask.mark.depends_on(PLOT_DEPENDENCIES)
@pytask.mark.parametrize("outcome, title, produces", PLOT_PARAMETRIZATION)
def task_plot_main_prediction_incidences(depends_on, outcome, title, produces):
    incidences = pd.read_pickle(depends_on["all_incidences"])
    to_plot = {key[0]: df for key, df in incidences.items() if key[1] == outcome}

    fig, ax = plot_incidences(
        incidences=to_plot,
        n_single_runs=15,
        title=title,
        name_to_label=NAME_TO_LABEL,
        rki=outcome,
    )
    ax.axvline(SCENARIO_START, label="Szenarienbeginn")
    fig.savefig(
        produces["fig"], dpi=200, transparent=False, facecolor="w", bbox_inches="tight"
    )


@pytask.mark.depends_on(PLOT_DEPENDENCIES)
@pytask.mark.produces({"b117": PREDICT_PATH / "virus_strain_shares_b117.png"})
def task_plot_main_prediction_virus_shares(depends_on, produces):
    strain_shares = pd.read_pickle(depends_on["all_strain_shares"])

    for strain, path in produces.items():
        to_plot = {scenario: df[strain] for scenario, df in strain_shares.items()}
        if strain != "base_strain":
            title = f"Anteil von {strain.title()} an den Infektionen"
        else:
            title = "Anteil des Corona Wildtyps an den Infektionen"

        fig, ax = plot_incidences(
            incidences=to_plot,
            n_single_runs=15,
            title=title,
            name_to_label=NAME_TO_LABEL,
            rki=False,
        )
        ax.axvline(SCENARIO_START, label="Szenarienbeginn")
        ax.set_ylabel("Anteil")

        empirical = pd.read_csv(depends_on["rki_strains"], parse_dates=["date"])
        empirical = empirical.set_index("date")
        start_date = SCENARIO_START - pd.Timedelta(days=35)
        empirical = empirical.loc[start_date:, f"share_{strain}"]

        sns.lineplot(
            x=empirical.index,
            y=empirical,
            label="Beobachteter Anteil",
            ax=ax,
            color="firebrick",
        )

        fig.savefig(
            path,
            dpi=200,
            transparent=False,
            facecolor="w",
            bbox_inches="tight",
        )
