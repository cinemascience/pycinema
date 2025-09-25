from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3, plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import pycinema
import pycinema.filters

import base64, io
from PIL import Image

# -----------------------------------------------------------------------------
# Trame setup
# -----------------------------------------------------------------------------
server = get_server()
state, ctrl = server.state, server.controller

state.images = []

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

    # Headers for vuetify3
    header_options = {h: {"title": h} for h in headers}

    state.headers, state.rows = vuetify3.dataframe_to_grid(df, header_options)

def rgba_to_base64(rgba):
    img = Image.fromarray(rgba[..., :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90, optimize=True, progressive=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode("utf-8")

@state.change("selected_rows")
def on_selection(**_):
    temp = []
    for i in state.selected_rows:
        for img in ImageReader_0.outputs.images.get():
            if img.meta["FILE"] == state.rows[i]["FILE"]:
                temp.append(
                    rgba_to_base64( img.channels["rgba"] )
                )
    state.images = temp

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
with SinglePageLayout(server) as layout:
    layout.title.set_text("PyCinema Trame Example")

    with layout.content:
        with vuetify3.VContainer(fluid=True):
            with vuetify3.VRow(dense=True,style='position:absolute; top:5em;bottom:10px;left:0;right:0'):
                with vuetify3.VCol(dense=True):
                    vuetify3.VTextField(
                            v_model=("sql", ""),
                            label='Table Query',
                            density='compact',
                            style='margin:1em',
                            hide_details=True,
                    )
                    vuetify3.VDataTable(
                        v_model=("selected_rows", []),
                        headers=("headers", []),
                        items=("rows", []),
                        classes="elevation-1 ma-4",
                        multi_sort=True,
                        density='compact',
                        items_per_page=10,
                        show_select=True,
                    )
                with vuetify3.VCol(dense=True, classes="fill-height", style='overflow:scroll;margin-left:1.5em;'):
                    vuetify3.VImg(
                        v_for="src, idx in images",
                        src=('images[idx]',),
                        style='margin-bottom:0.5em'
                    )
state.sql = "SELECT * FROM input LIMIT 10"

if __name__ == "__main__":
    server.start()

