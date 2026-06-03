#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "trame",
#   "trame-vuetify",
#   "trame-plotly",
#   "plotly",
#   "pandas",
#   "pillow",
#   "pycinema @ git+https://github.com/cinemascience/pycinema/@uv_test",
# ]
# ///

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import plotly, vuetify3, html
import plotly.graph_objects as go
import pandas as pd
import numpy

import matplotlib.pyplot as pp
import pycinema

import base64, io
from PIL import Image

server = get_server()
state, ctrl = server.state, server.controller
state.scale = 40
# state.database_url = 'https://raw.githubusercontent.com/cinemascience/pycinema/master/data/sphere.cdb'
state.database_url = 'https://raw.githubusercontent.com/cinemascience/pycinema/master/data/scalar-images.cdb'
state.axes = []

CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()
DepthCompositing_0 = pycinema.filters.DepthCompositing()
ColorMapping_0 = pycinema.filters.ColorMapping()
ShaderSSAO_0 = pycinema.filters.ShaderSSAO()
ImageAnnotation_0 = pycinema.filters.ImageAnnotation()

CinemaDatabaseReader_0.inputs.path.set(state.database_url, False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set('SELECT * from input LIMIT 0', False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
DepthCompositing_0.inputs.images_a.set(ImageReader_0.outputs.images, False)
DepthCompositing_0.inputs.compose.set(('object_id', {'p': 0, 's0': 1, 's1': 2}), False)
ColorMapping_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ColorMapping_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ShaderSSAO_0.inputs.images.set(ColorMapping_0.outputs.images, False)
ImageAnnotation_0.inputs.images.set(ShaderSSAO_0.outputs.images, False)

@state.change("database_url")
def update_database_url(**_):
    CinemaDatabaseReader_0.inputs.path.set(state.database_url)

@state.change("selected_rows")
def on_selection(**_):
    ids = []
    for s in state.selected_rows:
        row = state.rows[s]
        ids.append(str(row['id']))
    TableQuery_0.inputs.sql.set('SELECT * from input where id in ('+','.join(ids)+')')

@state.change("channel","colormap","scalar_min","scalar_max")
def update_channel(**_):
    ColorMapping_0.inputs.channel.set(state.channel,False)
    ColorMapping_0.inputs.map.set(state.colormap,False)
    ColorMapping_0.inputs.range.set((state.scalar_min,state.scalar_max),False)
    ImageAnnotation_0.update()

@state.change("scale")
def update_scale(**_):
    state.image_style = 'margin:0.25em;float:left;width:'+str(state.scale)+'%'

@state.change("quality")
def update_quality(**_):
    syncPipelineWithTrame(ImageAnnotation_0)

def update_axis_value_select(index, value):
    print(index,value)

def update_axis_value(index, value):
    if state.axes[index]["value"]==value:
        return
    print(index,value)
    axes = list(state.axes)
    axes[index] = {
        **axes[index],
        "value": value,
    }
    state.axes = axes

    sql = 'SELECT * FROM input WHERE '
    for a in axes:
      sql += '"'+a["name"]+'" == "'+str(a["values"][a["value"]])+'" AND '

    sql += ' '
    sql = sql[:-6]

    TableQuery_0.inputs.sql.set(sql)

def rgba_to_base64(rgba):
    img = Image.fromarray(rgba[..., :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(state.quality), optimize=True, progressive=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode("utf-8")

def syncPipelineWithTrame(filter):
    if filter==CinemaDatabaseReader_0:
        table = CinemaDatabaseReader_0.outputs.table.get()
        headers, *rows = table

        df = pd.DataFrame(rows, columns=headers)
        header_options = {h: {"title": h} for h in headers}
        state.headers, state.rows = vuetify3.dataframe_to_grid(df, header_options)

        header_axes = [h for h in headers if h.lower() not in ['id','file']]
        axes = []
        for h in header_axes:
            values = list(set([row[h] for row in state.rows]))
            isNumeric = values[0].isnumeric()
            if isNumeric:
                values = [float(v) for v in values]
            values.sort()
            axes.append({
                "name": h,
                "value": 0,
                "values": [str(v) for v in values],
                "isNumeric": isNumeric
            })
        state.axes = axes
        state.selected_rows = []

    elif filter==ImageAnnotation_0:
        images = ImageAnnotation_0.outputs.images.get();
        for img in images:
            state.channels = list(img.channels.keys())
            break

        state.images = [ rgba_to_base64( img.channels["rgba"] ) for img in images ]

pycinema.Filter.on('filter_updated',syncPipelineWithTrame)

with SinglePageLayout(server) as layout:
    layout.title.set_text("Table Images Demo")

    with layout.toolbar:
        vuetify3.VSpacer()
        vuetify3.VSlider(
            v_model=("scale", 100),
            style='max-width:15em;margin-top: 1em; margin-right:1em;',
            label='Image Scale',
            density='compact',
            min=0,
            max=100,
        )
        vuetify3.VSlider(
            v_model=("quality", 80),
            style='max-width:15em;margin-top: 1em; margin-right:1em;',
            label='Image Quality',
            density='compact',
            min=0,
            max=100,
        )
        vuetify3.VSelect(
            v_model=("channel", "rgba"),
            label='Field',
            items=("channels", ['rgba']),
            style='max-width:20em;margin-top: 1em; margin-right:1em;',
            density='compact',
        )
        vuetify3.VSelect(
            v_model=("colormap", 'plasma'),
            style='max-width:10em;margin-top: 1em; margin-right:1em;',
            label='Colormap',
            items=("colormaps", [map for map in pp.colormaps() if not map.endswith('_r')]),
            density='compact',
            disabled=("channel==='rgba'",)
        )
        vuetify3.VNumberInput(
            v_model=("scalar_min", 0),
            style='max-width:6em;margin-top: 1em; margin-right:1em;',
            label='Min',
            density='compact',
            control_variant='hidden',
            disabled=("channel==='rgba'",)
        )
        vuetify3.VNumberInput(
            v_model=("scalar_max", 1),
            style='max-width:6em;margin-top: 1em; margin-right:1em;',
            label='Max',
            density='compact',
            control_variant='hidden',
            disabled=("channel==='rgba'",)
        )

    with layout.content:
        with vuetify3.VRow():
            with vuetify3.VCol():
                with vuetify3.VRow():
                    with vuetify3.VCol():
                        vuetify3.VTextField(
                                v_model=("database_url", ""),
                                label='database url',
                                density='compact',
                                style='margin:1em',
                                hide_details=True,
                        )
                # with vuetify3.VRow():
                #     with vuetify3.VCol():
                #         with vuetify3.VContainer(
                #                     v_for="a,i in axes",
                #                     key=("a.name",),
                #                     style='',
                #                 ):
                #
                #                 with vuetify3.VRow():
                #                   with vuetify3.VCol():
                #                     vuetify3.VSlider(
                #                         v_if=("a.isNumeric",),
                #                         v_model=("a.value",),
                #                         update_modelValue=(update_axis_value, "[i, $event]"),
                #                         label=("a.name",),
                #                         min=0,
                #                         max=('a.values.length-1',),
                #                         step=1,
                #                     )
                #                     vuetify3.VSelect(
                #                         v_if=("!a.isNumeric",),
                #                         v_model=("a.value", ''),
                #                         label=('a.name',),
                #                         items=('a.values',),
                #                         density='compact',
                #                         update_modelValue=(update_axis_value_select, "[i, $event]"),
                #                     )
                #
                #                   with vuetify3.VCol():
                #                     vuetify3.VLabel('{{axes[i].values[axes[i].value]}}')
                with vuetify3.VRow():
                    with vuetify3.VCol():
                        vuetify3.VDataTable(
                            v_model=("selected_rows", []),
                            headers=("headers", []),
                            items=("rows", []),
                            classes="elevation-1 ma-4",
                            multi_sort=True,
                            dense=True,
                            items_per_page=10,
                            multi_select=True,
                            show_select=True,
                        )

            with vuetify3.VCol(dense=True, classes="fill-height", style='overflow:scroll;margin-left:1.5en;max-height:1200px'):
                vuetify3.VImg(
                    v_for="src, idx in images",
                    src=('images[idx]',),
                    style=('image_style','margin:0.25em;float:left;width:40%')
                )

ImageAnnotation_0.update()

if __name__ == "__main__":
    server.start()
