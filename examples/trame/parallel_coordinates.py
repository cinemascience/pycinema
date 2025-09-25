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

CinemaDatabaseReader_0 = pycinema.filters.CinemaDatabaseReader()
TableQuery_0 = pycinema.filters.TableQuery()
ImageReader_0 = pycinema.filters.ImageReader()
DepthCompositing_0 = pycinema.filters.DepthCompositing()
ColorMapping_0 = pycinema.filters.ColorMapping()
ShaderSSAO_0 = pycinema.filters.ShaderSSAO()
ImageAnnotation_0 = pycinema.filters.ImageAnnotation()

CinemaDatabaseReader_0.inputs.path.set("./data/scalar-images.cdb", False)
TableQuery_0.inputs.table.set(CinemaDatabaseReader_0.outputs.table, False)
TableQuery_0.inputs.sql.set('SELECT * from input LIMIT 0', False)
ImageReader_0.inputs.table.set(TableQuery_0.outputs.table, False)
DepthCompositing_0.inputs.images_a.set(ImageReader_0.outputs.images, False)
DepthCompositing_0.inputs.compose.set(('object_id', {'p': 0, 's0': 1, 's1': 2}), False)
ColorMapping_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ColorMapping_0.inputs.images.set(DepthCompositing_0.outputs.images, False)
ShaderSSAO_0.inputs.images.set(ColorMapping_0.outputs.images, False)
ImageAnnotation_0.inputs.images.set(ShaderSSAO_0.outputs.images, False)

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

def get_integers_from_ranges(ranges):
    return sorted({i for start, end in ranges for i in range(int(start) + (start % 1 > 0), int(end) + 1)})

def on_constraint(constraints):
    for dim_constraints in constraints:
        for dim, val in dim_constraints.items():
            idx = int(dim.split("[")[1].split("]")[0])
            state.dimensions[idx]['constraintrange'] = (
                numpy.squeeze(numpy.array(val)) if val is not None else []
            )
        break  # Only apply the first constraint block

    conditions = []

    for d in state.dimensions:
        ranges = numpy.atleast_2d(d['constraintrange'])
        if ranges.size == 0:
            continue

        if 'ticktext' in d:
            indices = get_integers_from_ranges(ranges)
            labels = [f'"{d["ticktext"][i]}"' for i in indices]
            conditions.append(f'"{d["label"]}" IN ({",".join(labels)})')
        else:
            range_clauses = [ f'"{d["label"]}" BETWEEN {r[0]} AND {r[1]}' for r in ranges ]
            conditions.append(f'({" OR ".join(range_clauses)})')

    TableQuery_0.inputs.sql.set(
        f'SELECT * from input WHERE {" AND ".join(conditions)}'
        if conditions else
        'SELECT * FROM input LIMIT 0'
    )

def rgba_to_base64(rgba):
    img = Image.fromarray(rgba[..., :3], mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(state.quality), optimize=True, progressive=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode("utf-8")

def syncPipelineWithTrame(filter):
    if filter==CinemaDatabaseReader_0:
        table = CinemaDatabaseReader_0.outputs.table.get()
        header = table[0]
        header_ = [i for i,h in enumerate(header) if h not in ['id','FILE']]
        rows = table[1:]

        data = {header[i]:[] for i in header_}
        for row in rows:
            for i in header_:
                data[header[i]].append(float(row[i]) if pycinema.isNumber(row[i]) else row[i])

        dimensions = []
        for key, values in data.items():
            first_val = values[0]
            if pycinema.isNumber(first_val):
                # Numeric axis
                dimensions.append(dict(
                    range=[min(values), max(values)],
                    label=key,
                    values=values,
                    constraintrange=[]
                ))
            else:
                # Categorical axis
                cats = {v: i for i, v in enumerate(sorted(set(values)))}
                dimensions.append(dict(
                    range=[0, len(cats)-1],
                    label=key,
                    values=[cats[v] for v in values],  # map strings to numbers
                    tickvals=list(cats.values()),
                    ticktext=list(cats.keys()),
                    constraintrange=[]
                ))

        state.dimensions = dimensions
        fig = go.Figure( data=go.Parcoords( dimensions=state.dimensions ) )
        ctrl.figure_update(fig)
    elif filter==TableQuery_0:
        table = TableQuery_0.outputs.table.get()
        headers, *rows = table

        df = pd.DataFrame(rows, columns=headers)
        header_options = {h: {"title": h} for h in headers}
        state.headers, state.rows = vuetify3.dataframe_to_grid(df, header_options)

    elif filter==ImageAnnotation_0:
        images = ImageAnnotation_0.outputs.images.get();
        for img in images:
            state.channels = list(img.channels.keys())
            break

        state.images = [ rgba_to_base64( img.channels["rgba"] ) for img in images ]

pycinema.Filter.on('filter_updated',syncPipelineWithTrame)

with SinglePageLayout(server) as layout:
    layout.title.set_text("Interactive Parallel Coordinates Demo")

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
                with vuetify3.VRow(dense=True):
                    with vuetify3.VCol():
                        figure = plotly.Figure(
                            display_logo=False,
                            display_mode_bar=False,
                            style='background-color:red; min-height:400px;min-width:200px',
                            restyle=(on_constraint, "[$event]"),
                        )
                        ctrl.figure_update = figure.update
                with vuetify3.VRow():
                    with vuetify3.VCol():
                        vuetify3.VDataTable(
                            v_model=("selected_rows", []),
                            headers=("headers", []),
                            items=("rows", []),
                            classes="elevation-1 ma-4",
                            density='compact',
                            items_per_page=10,
                            show_select=False,
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
