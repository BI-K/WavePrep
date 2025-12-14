# WavePrep
All scripts are configured to only take in config files as input. Any configuration is to be done in the config files present in the `configs/` directory.

There is a `configs/base_config.json` file which contains the base configuration like logging and output directory structure. Each script has its own folder in the `configs/` directory with a `default.json` file which contains the script-specific configurations.

This ReadMe wil still be updated.

## MIMIC-III Waveform Database Matched Subset

The dataset consists of 10,282 Patients and 22,247 numeric records. Only 10,269 patients have at least one numeric record.

## 1) Provide a .csv-file that conains the records_id, start_offset and end_offest of the recording time of each record you want to create a dataset of

File Structure:

|record | offset_start_seconds | offset_end_seconds |
|-------|----------------------| -------------------|
|p000020-2183-04-28-17-47n | 0 |78899.9999998422 |
|p000033-2116-12-24-12-35n | 0 | 80819.99999983836 |


## 2) Configure the Dataset Creation

## 3) Create Database Script

- `create_dataset.py`: This script creates the MIMIC-III Waveform Database Matched Subset.

You can run it with the command:
```bash
# To run the create script
python scripts/create_dataset.py --config configs/create/default.json
```

The `configs/create/default.json` file contains the following to configure:
1. The input_channel and the output_channel to create the database can be configured.
2. The `record_list_file`: The file containing the list of records to create the database. This has to be given or else the script will not run.
3. The signal_processing parameters have 2 configurations:
   - `desired_resolution`: The desired resolution of the signal in samples/second.
   - `downsampling_strategy`: The strategy to downsample the signal. Default is `decimate`, but configurable to `mean` and others.
4. The windowing or slicing parameters have 4 configurations:
    - `observation_window`: The length of the observation window in seconds. The observation window is the time period for which the signal is to be observed.
    - `prediction_horizon`: The length of the prediction horizon in seconds. The prediction horizon is the time after the observation window for which the signal is to be predicted.
    - `prediction_window`: The length of the prediction window in seconds. The prediction window is the time period for which the signal is to be predicted.
    - `step` - The step size in seconds for the sliding window. This is the time interval between two consecutive windows.
5. The validation parameters have 3 configurations:
    - `min_record_duration`: The minimum duration of the record in seconds. If the record is shorter than this, it will be skipped.
    - `validate_channels`: If set to `True`, the script will validate the channels in the record. If set to `False`, it will skip the validation.
    - `strict_nan_check`: If set to `True`, the script will check for NaN values in the record. If set to `False`, it will skip the NaN check.

The create dataset script will use the ThreadPoolExecutor to parallelize the creation of the records. The number of workers is set to 8 by default (which is fixed, it is an ideal number of workers for most systems).

Each create script run will create a new folder in the `outputs/` directory with the name of the current datetime. The folder will contain analysis, reports, logs, and mainly data folder which will have like folder names with the subject ID and the generated csv files using the records in the subject ID folder.

Splitting takes place by taking the subject IDs from the outputs/create/latest/data/ folder and then splitting them into train, validation, and test sets. The split is done using a "group shuffle" approach, which means that all records belonging to a single subject (group) are kept together in the same split (train, validation, or test). This ensures that the records of a subject are not divided across different sets, preventing data leakage between splits. For more information on group shuffle, see [GroupShuffleSplit in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).

The `configs/split/default.json` file contains the following to configure:
1. The `input_path`: The path to the input data folder. The default is `outputs/create/latest/data/`, which takes the latest created dataset.
2. The `train_ratio`, `validation_ratio`, and `test_ratio`: The ratios for splitting the dataset into train, validation, and test sets. The sum of these ratios should be equal to 1.0.
3. The `min_samples_per_subject`: The minimum number of samples required for a subject to be included in the split. If a subject has fewer samples than this, it will be skipped.
4. The `exclude_subjects` list: A list of subject IDs to be excluded from the split can be provided.
5. The `include_only_subjects` list: A list of subject IDs to be included in the split can be provided. If this is provided, only the subjects in this list will be included in the split.
6. The `dry_run`: If set to `True`, the copying of files will not be done, and only the split information will be printed. This is useful for testing the split without actually copying the files. If set to `False`, the files will be copied to the respective train, validation, and test folders.

The output folder is `outputs/split/` where a new folder will be created for every split run.
