from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3, plotly
import plotly.graph_objects as go
import pandas as pd
import pycinema
import pycinema.filters

# -----------------------------------------------------------------------------
# Trame setup
# -----------------------------------------------------------------------------
server = get_server()
state, ctrl = server.state, server.controller

# -----------------------------------------------------------------------------
# PyCinema pipeline
# -----------------------------------------------------------------------------
CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()

CinemaDatabaseReader_0.inputs.path.set("./data/sphere.cdb", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)

# -----------------------------------------------------------------------------
# Sync
# -----------------------------------------------------------------------------
@state.change("sql")
def on_sql_change(**_):
    TableQuery_0.inputs.sql.set(state.sql)
    ImageReader_0.update()

    table = TableQuery_0.outputs.table.get()
    headers, *rows = table

    # Build DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # Headers for Vuetify
    header_options = {h: {"title": h} for h in headers}

    state.headers, state.rows = vuetify3.dataframe_to_grid(df, header_options)

@state.change("selected_rows")
def on_selection(**_):
    if not state.selected_rows:
        fig = go.Figure(data=go.Image(z=[[(0,0,0)]]))
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            dragmode=False,
        )
        ctrl.figure_update(fig)
        return

    selection = state.rows[state.selected_rows[0]]
    for img in ImageReader_0.outputs.images.get():
        if img.meta["FILE"] == selection["FILE"]:
            rgba = img.channels["rgba"]
            fig = go.Figure(data=go.Image(z=rgba))
            fig.update_layout(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                dragmode=False,
            )
            ctrl.figure_update(fig)
            return

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
with SinglePageLayout(server) as layout:
    layout.title.set_text("PyCinema Trame Example")

    with layout.toolbar:
        vuetify3.VSpacer()
        vuetify3.VTextField(
            v_model=("sql", ""),
            dense=True,
            hide_details=True,
        )

    with layout.content:
        with vuetify3.VContainer(fluid=True):
            with vuetify3.VRow(dense=True):
                with vuetify3.VCol(dense=True):
                    vuetify3.VDataTable(
                        v_model=("selected_rows", []),
                        headers=("headers", []),
                        items=("rows", []),
                        classes="elevation-1 ma-4",
                        multi_sort=True,
                        dense=True,
                        items_per_page=10,
                        multi_select=False,
                        show_select=True,
                    )
                with vuetify3.VCol(dense=True, classes="fill-height"):
                    figure = plotly.Figure(
                        display_logo=False,
                        display_mode_bar=False
                    )
                    ctrl.figure_update = figure.update

state.sql = "SELECT * FROM input LIMIT 10"

if __name__ == "__main__":
    server.start()

