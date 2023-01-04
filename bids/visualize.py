from __future__ import annotations

import math
import warnings
from pathlib import Path
import re
from typing import Any

from bids import BIDSLayout
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class EventPlotter:
    """Class to handle plotting the content of events files.

    Parameters
    ----------
    events_file : str | Path
        Path to the events file.

    event_column : str | None, optional
        Name of the column containing the events.
        If `None`, events types are assumed to be in the "trial_type" column.

    include : list[str] | None, optional
        List of events type to include in the plot.
    """

    def __init__(
        self,
        events_file: str | Path,
        event_column: str | None = None,
        include: list[str] | None = None,
    ):
        self.COLOR_LIST = px.colors.qualitative.Plotly
        self.FONT_SIZE = 14
        self.STANDOFF = 16
        self.LINE_WIDTH = 3
        self.GRID_COLOR = "black"
        self.AXES_COLOR = "black"
        self.AXES_LINE_WIDTH = 2
        self.BG_COLOR = "rgb(255,255,255)"
        self.EVENT_HEIGHT = 1
        self.FAST_RESPONSE_THRESHOLD = 0.2
        self.NB_BINS = 40
        self.TWO_COLUMN_WIDTHS = [0.7, 0.3]
        self.THREE_COLUMN_WIDTHS = [0.7, 0.15, 0.15]
        self.FOUR_COLUMN_WIDTHS = [0.7, 0.1, 0.1, 0.1]
        self.TICK_LENGTH = 6

        self.title: None | str = None
        self.fig = None

        self._trial_type_index: int = 0
        self._trial_types: None | list[str] = None
        self._bottom_row: dict(str, list[int]) = {
            "ISI": [],
            "duration": [],
            "response": [],
        }

        self._get_data_from_file(events_file)

        self.event_column = event_column
        if event_column is None:
            self.event_column = "trial_type"

        if len(self.data.columns.to_list()) == 2:
            warnings.warn(
                f"""Only columns 'onset' and 'duration' found in:
    {events_file}
Creating a dummy trial_type column.
            """
            )
            self.data["trial_type"] = "trial_type"

        self.trial_types(include=include)
        if self.trial_types() is None or len(self.trial_types()) == 0:
            warnings.warn(
                f"No trial types found in {events_file} for 'include={include}'"
            )
            return

        self.fig = go.FigureWidget(
            make_subplots(
                rows=len(self.trial_types()),
                cols=self.nb_cols,
                horizontal_spacing=0.02,
                vertical_spacing=0.08,
                shared_xaxes=True,
                column_widths=self.column_widths,
            )
        )

    @property
    def nb_trial_types(self) -> int:
        return len(self.trial_types())

    @property
    def nb_cols(self) -> int:
        value = 2
        if self.plot_duration_flag:
            value += 1
        if "response_time" in self.data.columns:
            value += 1
        return value

    @property
    def plot_duration_flag(self) -> bool:
        """Do not plot duration if all durations are the same"""
        if get_duration(self.data).unique().size == 1:
            return False

        idx = self._trial_type_index
        tmp = [
            get_duration(self.data_this_trial_type).unique().size > 1
            for self._trial_type_index in range(self.nb_trial_types)
        ]
        self._trial_type_index = idx
        return any(tmp)

    @property
    def column_widths(self) -> list[float]:
        if self.nb_cols == 2:
            return self.TWO_COLUMN_WIDTHS
        elif self.nb_cols == 3:
            return self.THREE_COLUMN_WIDTHS
        else:
            return self.FOUR_COLUMN_WIDTHS

    """Properties that are specific to a given trial type."""

    @property
    def this_trial_type(self) -> str:
        return self.trial_types()[self._trial_type_index]

    @property
    def this_color(self) -> str:
        return self.COLOR_LIST[self._trial_type_index]

    @property
    def this_row(self) -> int:
        subplot_rows = list(range(1, self.nb_trial_types + 1))
        return subplot_rows[self._trial_type_index]

    @property
    def data_this_trial_type(self) -> pd.DataFrame:
        mask = self.data[self.event_column] == self.this_trial_type
        return self.data[mask]

    def trial_types(self, include: list[str] = None) -> list[str]:
        """Set trial types that will be plotted."""
        if self._trial_types is not None:
            return self._trial_types

        if self.event_column in self.data.columns:
            trial_type = self.data[self.event_column]
            trial_type.dropna(inplace=True)
            self._trial_types = trial_type.unique()
        else:
            warnings.warn(f"No column '{self.event_column}' in {self.title}")
            return
        if include is not None:
            self._trial_types = list(set(self._trial_types) & set(include))

    def _get_data_from_file(self, events_file: str | Path) -> None:
        events_file = Path(events_file)
        if not events_file.exists():
            raise FileNotFoundError(f"File {events_file} does not exist.")

        self.title = events_file.name
        self.data = pd.read_csv(events_file, sep="\t")

    # Used only to keep track on which row to plot the x axis title
    # for the histograms.
    def bottom_row(
        self, col_name: str | None = None, new_value: int | None = None
    ) -> int | None:
        if col_name is None:
            col_name = "ISI"

        if new_value is None:
            return (
                None
                if len(self._bottom_row[col_name]) == 0
                else self._bottom_row[col_name][0]
            )

        self._bottom_row[col_name].append(new_value)
        self._bottom_row[col_name] = [max(self._bottom_row[col_name])]
        return None

    """Wrapper methods"""

    def plot(self) -> None:
        """Plot trial types.
        """
        self.plot_trial_types()
        self._update_axes()

    def show(self) -> None:
        """Show the figure.
        """
        self.fig.show()

    """Plotting methods"""

    def plot_trial_types(self) -> None:
        """Loop over trial types and plot them one by one.

        Plots the following:
        - trial type timeline
        - ISI histogram
        - duration histogram
        - response time timeline if response_time is present
        - response time histogram if response_time is present
        """

        for self._trial_type_index in range(self.nb_trial_types):

            onset = self.data_this_trial_type["onset"]
            duration = get_duration(self.data_this_trial_type)

            x = np.array([[onset], [onset + duration]]).flatten("F")
            y = np.tile([self.EVENT_HEIGHT, 0], (1, len(x) // 2))[0]

            self._plot_timeline(
                x,
                y,
                name=self.this_trial_type,
                mode="lines",
                color=self.this_color,
            )

            self._plot_responses()

            self._default_axes(col=1)

            self.fig.update_yaxes(
                row=self.this_row,
                col=1,
                showticklabels=False,
                ticklen=0,
            )

            isi = onset.diff()

            status = self._plot_histogram(
                isi,
                col=2,
                prefix="ISI",
            )
            self.bottom_row("ISI", status)

            if self.plot_duration_flag:
                status = self._plot_histogram(
                    duration,
                    col=3,
                    prefix="duration",
                )
                self.bottom_row("duration", status)

    def _plot_timeline(self, x: Any, y: Any, name: str, mode: str, color: str) -> None:

        x = np.append(0, x)
        y = np.append(0, y)

        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                line_shape="hv",
                line_width=self.LINE_WIDTH,
                mode=mode,
                line_color=color,
                legendgroup=str(self.this_row),
                legendgrouptitle_text=f"Group - {self.this_trial_type}",
            ),
            row=self.this_row,
            col=1,
        )
        self.fig.update_yaxes(
            range=[0, self.EVENT_HEIGHT + 0.1],
            row=self.this_row,
            col=1,
        )
        # add ticks every 15 and 60 seconds
        self.fig.update_xaxes(
            row=self.this_row,
            col=1,
            dtick=60,
            minor=dict(
                dtick=15,
                ticklen=self.TICK_LENGTH / 2,
            ),
        )

    def _plot_responses(self) -> None:

        if "response_time" not in self.data.columns:
            return

        self._plot_response_timeline(
            self.data_this_trial_type["response_time"],
            self.data_this_trial_type["onset"],
        )

        mask = self.data_this_trial_type["response_time"] < self.FAST_RESPONSE_THRESHOLD
        if mask.any():
            fast_response_time = self.data_this_trial_type["response_time"][mask]
            fast_response_onset = self.data_this_trial_type["onset"][mask]
            self._plot_response_timeline(
                fast_response_time,
                fast_response_onset,
                prefix="fast responses",
                color="red",
            )

        status = self._plot_histogram(
            self.data_this_trial_type["response_time"],
            col=self.nb_cols,
            prefix="response time",
        )
        self.bottom_row("response", status)

    def _plot_response_timeline(
        self,
        response_time: pd.Series,
        onset: pd.Series,
        prefix: str | None = None,
        color: str = "black",
    ) -> None:

        if prefix is None:
            prefix = "responses"
        name = f"{prefix} {self.this_trial_type}"

        responses_onset = onset + response_time
        responses_onset.dropna(inplace=True)

        x = np.array([[responses_onset], [responses_onset]]).flatten("F")
        y = np.tile([self.EVENT_HEIGHT / 2, 0], (1, len(x) // 2))[0]

        self._plot_timeline(
            x,
            y,
            name=name,
            mode="lines+markers",
            color=color,
        )

    def _plot_histogram(
        self,
        values: pd.Series,
        col: int,
        prefix: str = "",
    ) -> None | int:
        mask = values.isnull() | values.isna()
        values.loc[mask] = 0
        if (values == 0).all() or values.unique().size == 1:
            return None

        self.fig.add_trace(
            go.Histogram(
                x=values,
                name=f"{prefix} {self.this_trial_type}",
                marker_color=self.this_color,
                nbinsx=self.NB_BINS,
                xbins=dict(  # bins used for histogram
                    start=0,
                ),
                legendgroup=str(self.this_row),
                legendgrouptitle_text=f"Group - {self.this_trial_type}",
            ),
            row=self.this_row,
            col=col,
        )

        self._default_axes(col)
        hist, bin_edges = np.histogram(values, bins=self.NB_BINS)
        self.fig.update_yaxes(
            row=self.this_row, col=col, dtick=math.ceil(max(hist) / 4)
        )

        # we keep track of the lowest row to plot the title of that column
        return self.this_row

    """Axis formatting methods"""

    def _default_axes(self, col: int) -> None:
        self.fig.update_yaxes(
            row=self.this_row,
            col=col,
            tickfont=dict(size=self.FONT_SIZE),
            ticklen=self.TICK_LENGTH,
            ticks="outside",
            tickwidth=self.AXES_LINE_WIDTH,
            tickcolor=self.AXES_COLOR,
            showline=True,
            linewidth=self.AXES_LINE_WIDTH,
            linecolor=self.AXES_COLOR,
            showticklabels=True,
        )

        self.fig.update_xaxes(
            row=self.this_row,
            col=col,
            tickfont=dict(size=self.FONT_SIZE),
            ticklen=self.TICK_LENGTH,
            ticks="outside",
            tickwidth=self.AXES_LINE_WIDTH,
            tickcolor=self.AXES_COLOR,
            showline=True,
            linewidth=self.AXES_LINE_WIDTH,
            linecolor=self.AXES_COLOR,
            showticklabels=True,
            autorange=True,
        )

    def _update_axes(self) -> None:

        self.fig.update_xaxes(
            row=self.nb_trial_types,
            col=1,
            title=dict(
                text="Time (s)",
                standoff=self.STANDOFF,
                font=dict(size=self.FONT_SIZE + 2),
            ),
        )

        self.label_axes_histogram(col=2, row=self.bottom_row("ISI"), text="ISI (s)")
        if self.plot_duration_flag:
            self.label_axes_histogram(
                col=3, row=self.bottom_row("duration"), text="duration (s)"
            )

        if "response_time" in self.data.columns:
            self.label_axes_histogram(
                col=self.nb_cols,
                row=self.bottom_row("response"),
                text="Response time (s)",
            )

        self.fig.update_layout(
            plot_bgcolor=self.BG_COLOR,
            paper_bgcolor=self.BG_COLOR,
            legend=dict(
                title_text="trial types",
                y=1,
                font_size=self.FONT_SIZE,
                groupclick="toggleitem",
            ),
            title=dict(
                text=f"<b>{self.title}<b>",
                x=0.025,
                y=0.98,
                font=dict(size=self.FONT_SIZE + 4),
            ),
            margin=dict(t=50, b=30, l=30, r=30, pad=0),
        )

    def label_axes_histogram(self, col: int, row: int | None, text: str) -> None:
        if row is None:
            row = self.nb_trial_types
        self.fig.update_xaxes(
            row=row,
            col=col,
            title=dict(
                text=text, standoff=self.STANDOFF, font=dict(size=self.FONT_SIZE + 2)
            ),
        )


def get_duration(df: pd.DataFrame) -> pd.Series:
    tmp = df.copy()
    mask = df["duration"].isnull()
    tmp.loc[mask, "duration"] = 0
    return tmp["duration"]


class LayoutPlotter:
    """Class to handle plotting the content of a BIDSLayout.

    Parameters
    ----------
    layout : BIDSLayout
        Define extra plot to be done by splitting the dataset by entities,
        defaults to `None`

    filters : dict, optional
        Any optional key/values to filter the layout on.
        Keys are entity names, values are regexes to filter on. For
        example, passing filters={'subject': 'sub-[12]'} would return
        only files that match the first two subjects.
    """

    def __init__(
        self,
        layout: BIDSLayout,
        filters: dict[str, list[str]] | None = None,
    ) -> None:

        self.FONT_SIZE = 16

        self.datatype = sorted(layout.get(return_type="id", target="datatype"))
        self.task = sorted(layout.get_task())
        self.title = layout.description.get("Name", "BIDS dataset")
        self.df_layout = (
            layout.to_df(**filters) if filters is not None else layout.to_df()
        )
        if len(self.datatype) > 0:
            self.df_layout.dropna(subset=["datatype"], inplace=True)

    def plot(
        self,
        plot_by: None | str | list[str] = None,
        output_dir: str | Path | None = None,
        show: bool = True,
    ) -> None:
        """Create summary figures of bids dataset content.

        Parameters
        ----------
        plot_by : None | str | list[str], optional
            Define extra plot to be done by splitting the dataset by entities,
            defaults to `None`

        output_dir: output_dir: str | Path | None, optional
            Where the figure should be saved, defaults to `None`

        show: bool, optional
            Set to `True` to show the output figures after plotting, defaults to `True`

        Subplot in each figure have subject as rows.

        If a session level is present, subplots are grouped by session along rows.

        - show number of files per subject for each datatype
        - show number of files per subject for each task
        - show number of files per subject for each entity passed in plot_by
        """

        self.plot_by_datatype(output_dir=output_dir, show=show)

        self.plot_by_task(output_dir=output_dir, show=show)

        if plot_by is not None:
            if isinstance(plot_by, str):
                plot_by = [plot_by]
            for entity in plot_by:
                self.plot_by_entity(entity=entity, output_dir=output_dir, show=show)

    def plot_by_datatype(
        self, output_dir: str | Path | None = None, show: bool = True
    ) -> px.density_heatmap:
        """Plot dataset content split by datatype.

        Parameters
        ----------
        output_dir: str | Path | None, optional
            Where the figure should be saved, defaults to ``None``

        show: bool, optional
            Set to `True` to show the output figures after plotting, defaults to `True`

        Returns
        -------
        px.density_heatmap
            Plotly figure

        """

        if len(self.datatype) > 0:

            if "session" in self.df_layout.columns.tolist():
                fig = self._generate_heat_map(
                    self.df_layout, x="datatype", y="subject", facet_row="session"
                )
            else:
                fig = self._generate_heat_map(self.df_layout, x="datatype", y="subject")

            self._set_axis_and_title(fig, suffix="datatype")

            if show:
                fig.show()

            self._write_html(fig, suffix="datatype", output_dir=output_dir)

    def plot_by_task(
        self, output_dir: str | Path | None = None, show: bool = True
    ) -> px.density_heatmap:
        """Plot dataset content split by task.

        Parameters
        ----------
        output_dir: str | Path | None, optional
            Where the figure should be saved, defaults to ``None``

        show: bool, optional
            Set to `True` to show the output figures after plotting, defaults to `True`

        Returns
        -------
        px.density_heatmap
            Plotly figure
        """
        if len(self.task) > 0:
            fig = self.plot_by_entity(entity="task", output_dir=output_dir, show=show)
            return fig

    def plot_by_entity(
        self,
        entity: str | None,
        output_dir: str | Path | None = None,
        show: bool = True,
    ) -> px.density_heatmap:
        """_summary_

        Parameters
        ----------
        entity: str | None
            Define extra plot to be done by splitting the dataset by entities,
            defaults to ``None``

        output_dir: str | Path | None, optional
            Where the figure should be saved, defaults to ``None``

        show: bool, optional
            Set to `True` to show the output figures after plotting, defaults to `True`

        Returns
        -------
        px.density_heatmap
            Plotly figure
        """
        if entity is None:
            return

        if entity not in self.df_layout.columns.tolist():
            warnings.warn(
                f"""Entity '{entity}' not found in layout.
    Entities available: {self.df_layout.columns.tolist()}"""
            )
            entity = None
        if entity is None:
            return

        df = self.df_layout.dropna(subset=[entity])

        facet_col = "datatype" if "datatype" in df.columns.tolist() else None

        if "session" in df.columns.tolist():
            fig = self._generate_heat_map(
                df, x=entity, y="subject", facet_row="session", facet_col=facet_col
            )
        else:
            fig = self._generate_heat_map(
                df, x=entity, y="subject", facet_col=facet_col
            )

        self._set_axis_and_title(fig, suffix=entity)

        if show:
            fig.show()

        self._write_html(fig, suffix=entity, output_dir=output_dir)

        return fig

    def _generate_heat_map(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        facet_col: str | None = None,
        facet_row: str | None = None,
    ) -> px.density_heatmap:
        fig = px.density_heatmap(
            df,
            x=x,
            y=y,
            facet_row=facet_row,
            facet_col=facet_col,
            facet_row_spacing=0.04,
            facet_col_spacing=0.04,
            color_continuous_scale="gray",
            text_auto=True,
        )
        return fig

    def _set_axis_and_title(self, fig: px.density_heatmap, suffix: str) -> None:
        fig.update_xaxes(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            tickfont=dict(size=self.FONT_SIZE),
            tickcolor="rgba(0,0,0,0)",
            ticklen=4,
            ticks="outside",
        )
        fig.update_yaxes(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            tickprefix="sub-",
            tickfont=dict(size=self.FONT_SIZE),
            tickcolor="rgba(0,0,0,0)",
            ticklen=4,
            ticks="outside",
        )
        fig.update_layout(
            title=dict(
                text=f"<b>{self.title}: {suffix}</b>",
                x=0.05,
                y=0.98,
                font=dict(size=self.FONT_SIZE + 2),
            )
        )

    def _fig_name(self, suffix: str) -> str:
        dataset_name = self.title.lower().replace(" ", "_")
        dataset_name = re.sub("[^0-9a-zA-Z]+", "", dataset_name)
        return f"dataset-{dataset_name}_splitby-{suffix}_summary.html"

    def _write_html(
        self, fig: px.density_heatmap, suffix: str, output_dir: str | Path | None = None
    ) -> None:
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_dir.joinpath(self._fig_name(suffix=suffix)))
