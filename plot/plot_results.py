import optuna
import sqlite3
import polars as pl
import tikzplotlib

# Segmentation results

# Connect to the SQLite database
database_path = 'segmentation_analysis.db'
conn = sqlite3.connect(database_path)

# Query to get the list of studies
query_studies = "SELECT study_id, study_name FROM studies;"
studies_df = pl.read_database(query_studies, conn)

# Segmentation: Two tank without substitution
study_name = 'two_tank_nsub'
study = optuna.load_study(study_name=study_name, storage='sqlite:///' + database_path)
valid_trials = [trial for trial in study.trials if trial.value is not None and all(v is not None for v in trial.params.values())]
filtered_study = optuna.create_study(study_name=study.study_name, storage='sqlite:///recovered_segmentation_analysis_2', load_if_exists=True)
filtered_study.add_trials(valid_trials)

fig = optuna.visualization.matplotlib.plot_slice(filtered_study, params=["start_width"])
tikzplotlib.save("seg_two_tank_nsub_obj_vs_startwidth.tex")
fig = optuna.visualization.matplotlib.plot_slice(filtered_study, params=["step_width"])
tikzplotlib.save("seg_two_tank_nsub_obj_vs_stepwidth.tex")

# Segmentation: Two tank with substitution
study_name = 'two_tank_sub'
study = optuna.load_study(study_name=study_name, storage='sqlite:///' + database_path)
valid_trials = [trial for trial in study.trials if trial.value is not None and all(v is not None for v in trial.params.values())]
filtered_study = optuna.create_study(study_name=study.study_name, storage='sqlite:///recovered_segmentation_analysis_2', load_if_exists=True)
filtered_study.add_trials(valid_trials)

fig = optuna.visualization.matplotlib.plot_slice(filtered_study, params=["saturation"])
tikzplotlib.save("seg_two_tank_sub_obj_vs_saturation.tex")
fig = optuna.visualization.matplotlib.plot_contour(filtered_study, params=["start_iterations", "step_iterations"])
tikzplotlib.save("seg_two_tank_sub_obj_vs_startstepiterations.tex")

# Grouping results
# Grouping: converter

database_path = 'grouping_analysis.db'
conn = sqlite3.connect(database_path)
query_studies = "SELECT study_id, study_name FROM studies;"
studies_df = pl.read_database(query_studies, conn)
study_name = 'converter'
study = optuna.load_study(study_name=study_name, storage='sqlite:///' + database_path)
valid_trials = [trial for trial in study.trials if trial.value is not None and all(v is not None for v in trial.params.values())]
filtered_study = optuna.create_study(study_name=study.study_name, storage='sqlite:///recovered_grouping_analysis_2', load_if_exists=True)
filtered_study.add_trials(valid_trials)

fig = optuna.visualization.matplotlib.plot_slice(filtered_study, params=["criterion-factor"])
tikzplotlib.save("group_converter_obj_vs_criterion.tex")
fig = optuna.visualization.matplotlib.plot_contour(filtered_study, params=["iterations", "populations"])
tikzplotlib.save("group_converter_obj_vs_iter-pop.tex")
