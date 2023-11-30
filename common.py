
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from pydantic import validator
from scipy.stats._distn_infrastructure import rv_discrete_frozen, rv_continuous_frozen



@dataclass
class FilterColumn:
    """
    Specify the column to filter records on.
    
    Attributes:
        name: Name of the column.
        ratio: Ratio of the values to keep. Must be between 0 (excluded) and 1 (included).
    """
    name: str
    ratio: float

    def __post_init__(self):
        if self.ratio < 0 or self.ratio >= 1:
            raise ValueError("Ratio must be between 0 (excluded) and 1 (included).")
@dataclass
class SamplingDistributionConfig:
    """A base class for configuring distributions."""

    ref: str
    params: Optional[Union[Dict, List]] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class ColumnConfig:
    """
    Configure the columns of a generative process.
    Attributes:
        name: Name of the column.
        distribution: Distribution to use for the column values.
        filter_column: Column to filter on. If None, no filter is applied.
    """
    time: str
    target: str
    anomaly_flag: str = "m_bool"
    filter_column: Optional[Union[FilterColumn, List[FilterColumn]]] = None

    def __post_init__(self):
        if isinstance(self.filter_column, FilterColumn):
            self.filter_column = [self.filter_column]
        elif isinstance(self.filter_column, list):
            for col in self.filter_column:
                if not isinstance(col, FilterColumn):
                    raise ValueError("Filter column must be of type FilterColumn or List[FilterColumn].")
        elif self.filter_column is not None:
            raise ValueError("Filter column must be of type FilterColumn or List[FilterColumn].")


@dataclass
class AnomalyConfig:
    """
    Configure the anomly generation process.
    Attributes:
        start_date: Start date of the anomalious period.
        end_date: End date of the anomalious period.
        num_events: Number of anomalious events to generate. Can be either fixed or given as a discrete frozen distribution.
        event_window_in_hours: Window size for each event in hours. Can be either fixed or given as a discrete frozen distribution. 0 means that the event will not be completed.
    """

    start_date: datetime
    end_date: datetime
    num_events: Union[int, rv_discrete_frozen, str, SamplingDistributionConfig]
    event_window_in_hours: Union[int, rv_discrete_frozen, str, SamplingDistributionConfig]
    column: ColumnConfig
    noise_level: Union[float, rv_continuous_frozen, str, SamplingDistributionConfig] = 0.1
    description: Optional[str] = None

    @validator("num_events", "event_window_in_hours", "noise_level", always=True)
    def validate_distribution(cls, value):
        if isinstance(value, str):
            return get_reference(value)

        if isinstance(value, SamplingDistributionConfig):
            if isinstance(value.params, dict):
                return get_reference(value.ref)(**value.params)
            elif isinstance(value.params, list):
                return get_reference(value.ref)(*value.params)
            else:
                return get_reference(value.ref)()

        return value

    def __post_init__(self):
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date.")

        if isinstance(self.num_events, int) and isinstance(self.event_window_in_hours, int):
            if self.num_events <= 0:
                raise ValueError("Number of events must be greater than 0.")
            if self.event_window_in_hours < 0:
                raise ValueError("Event window size must be greater than or equal to 0.")
            if self.num_events > 1 and self.event_window_in_hours == 0:
                raise ValueError("Event window size must be greater than 0 if more than one event is generated.")




def generate_noise_events(dataset: pd.DataFrame, config: AnomalyConfig, inplace: bool = False):
    """Generate the noise events in the given dataset"""

    dataset = dataset.copy() if not inplace else dataset

    # Get the target and time column
    target_col = config.column.target
    noisy_target_col = f"m_{target_col}"
    noisy_flag = "m_bool"
    time_col = config.column.time

    # Iterate through the number of events
    num_events = config.num_events if isinstance(config.num_events, int) else config.num_events.rvs(1)[0]
    random_timepoints = np.random.randint(0, (config.end_date - config.start_date).total_seconds() // 3600,
                                          size=num_events)
    for t_point in random_timepoints:
        # Get the start date of the event randomly between the start and end date
        start_event_time = config.start_date + pd.Timedelta(hours=t_point)
        # Get the window size for the event
        window_size = config.event_window_in_hours if isinstance(config.event_window_in_hours, int) else \
            config.event_window_in_hours.rvs(1)[0]
        # Get the end time of the event
        end_event_time = start_event_time + pd.Timedelta(hours=window_size) if window_size > 0 else config.end_date
        # Get the event data
        event_data = dataset[(dataset[time_col] >= start_event_time) & (dataset[time_col] <= end_event_time)]
        # Get the noise level
        noise_level = config.noise_level if isinstance(config.noise_level, float) else config.noise_level.rvs(1)[0]
        # Add noise to the new target column
        dataset.loc[event_data.index, noisy_target_col] = dataset.loc[event_data.index, target_col] + np.random.normal(
            loc=0, scale=noise_level * dataset[target_col].max(skipna=True), size=len(event_data))
        # Set the flag to True
        dataset.loc[event_data.index, noisy_flag] = True

    return dataset


# Create functions to add noise/anomalies based on the anomaly configurattion
def add_noise(dataset: pd.DataFrame, config: AnomalyConfig, inplace: bool = False):
    """Add noise to the target column."""

    if not inplace:
        dataset = dataset.copy()

    target_col = config.column.target
    noisy_target_col = f"m_{target_col}"
    noisy_flag = "m_bool"

    # Add new column
    dataset.loc[:, noisy_target_col] = dataset.loc[:, target_col]
    # Set the flag to False
    dataset.loc[:, noisy_flag] = False

    # Get the filter columns
    filter_cols = config.column.filter_column
    if filter_cols is not None:
        filtered_data = dataset.copy()
        # Filter the data based on the filter columns
        for col in filter_cols:
            # Filter the data based on the specified column and ratio of randomly selected values
            # Select random unique values based on the ratio
            values = filtered_data[col.name].unique()
            num_values = int(col.ratio * len(values))
            values = np.random.choice(values, num_values, replace=False)
            # Filter the data
            filtered_data = filtered_data[filtered_data[col.name].isin(values)]

        # Apply the noise generation process
        noised_data = filtered_data.groupby(by=[f_col.name for f_col in filter_cols]).apply(generate_noise_events,
                                                                                            config=config)

        # Update the data with the noised data
        noised_data.reset_index(drop=True, inplace=True)
        dataset.loc[filtered_data.index, noisy_target_col] = noised_data[noisy_target_col].values
        dataset.loc[filtered_data.index, noisy_flag] = noised_data[noisy_flag].values
    else:
        noised_data = generate_noise_events(dataset, config)
        dataset.loc[:, noisy_target_col] = noised_data[noisy_target_col].values
        dataset.loc[:, noisy_flag] = noised_data[noisy_flag].values


    # Set negative values to 0
    dataset.loc[dataset[noisy_target_col] < 0, noisy_target_col] = 0.0
    if not inplace:
        return dataset




