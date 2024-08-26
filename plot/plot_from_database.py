import sqlite3
import polars as pl
import matplotlib.pyplot as plt
import optuna

# Connect to the SQLite database
database_path = 'recovered_segmentation_analysis.db'
conn = sqlite3.connect(database_path)


# Query to get the list of studies
query_studies = "SELECT study_id, study_name FROM studies;"
studies_df = pl.read_database(query_studies, conn)
study_map = {row['study_id']: row['study_name'] for row in studies_df.to_dicts()}

# Print the list of studies
print(studies_df)

for study_id, study_name in study_map.items():
    print(f"Study ID: {study_id}, Study Name: {study_name}")

    # Query trial data
    query_trials = f"""
    SELECT trial_id, study_id, state
    FROM trials
    WHERE state = 'COMPLETE' AND study_id = {study_id};
    """
    trials_df = pl.read_database(query=query_trials, connection=conn)

    # Query parameter data
    query_params = f"""
    SELECT trial_id, param_name, param_value
    FROM trial_params
    WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = {study_id} AND state = 'COMPLETE');
    """
    params_df = pl.read_database(query=query_params, connection=conn)

    # Query objective value
    query_objective = f"""
    SELECT trial_id, value
    FROM trial_values
    WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = {study_id} AND state = 'COMPLETE');
    """
    objective_df = pl.read_database(query=query_objective, connection=conn)

    # Merge trial and parameter data
    data = trials_df.join(params_df, on='trial_id')
    data = data.join(objective_df, on='trial_id')

    print(data.head())

    # Sort by trial_id for plotting
    data.sort('trial_id')

    # Plot objective value vs. trial
    plt.figure(figsize=(10, 6))
    plt.plot(data['trial_id'], data['value'], marker='o', linestyle='-', label='Objective Value')
    plt.xlabel('Trial ID')
    plt.ylabel('Objective Value')
    plt.title(f'Objective Value vs. Trial ID for Study: {study_name}')
    plt.grid(True)
    plt.legend()
    plt.show()

# Close the database connection
conn.close()

# Calculate Parameter Importance

for study_id, study_name in study_map.items():
    # Load the study
    study = optuna.load_study(study_name=study_name, storage='sqlite:///' + database_path)

    # Compute parameter importances
    importances = optuna.importance.get_param_importances(study)

    # Print importances
    print("Parameter importances:", study_name)
    for param, importance in importances.items():
        print(f"{param}: {importance:.4f}")