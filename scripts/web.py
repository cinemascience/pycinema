#!/usr/bin/env python

# import asyncio
# from websockets.asyncio.server import serve

import json
import os
import asyncio
from aiohttp import web
import websockets
from websockets.asyncio.server import serve

import zlib
import base64

import pycinema
import pycinema.filters

# pycinema settings
PYCINEMA = { 'VERSION' : '2.1.0'}

# this application settings
VIEW = { 'VERSION' : '1.0'}

# reporting
print("view v" + VIEW["VERSION"])


filter_list = dict([(name, cls) for name, cls in pycinema.filters.__dict__.items() if isinstance(cls,type) and issubclass(cls,pycinema.Core.Filter) and len(cls.__subclasses__())<1])

def get_by_path(obj, path):
    for key in path:
        if isinstance(obj, dict):
            obj = obj[key]
        elif isinstance(obj, list):
            obj = obj[int(key)]
        elif hasattr(obj,key):
            obj = getattr(obj,key)
        else:
            raise TypeError(f"Invalid path element '{key}' for type {type(obj)}")
    return obj

websocket_list = []
async def echo(websocket):
  print('connected')

  websocket_list.append(websocket)

  i = 0
  async for message_raw in websocket:
    message = json.loads(message_raw)
    # print(i,message)
    i+=1
    if message['header'] == 'get_filter_list':
      await send_message('filter_list',[*filter_list],message['id'])

    elif message['header'] == 'get_filters':
      await send_message('filter_list',[f.toJSON() for f in pycinema.Core.Filter._filters],message['id'])

    elif message['header'] == 'get_port_value':
      filter_id = message['payload'][0]
      port_name = message['payload'][1]
      path = message['payload'][2]
      f = [f for f in pycinema.Core.Filter._filters if f.id==filter_id][0]
      arr = get_by_path(
        f.outputs.get(port_name).get(),
        path
      )

      value_b64 = {
        'data': base64.b64encode(arr.tobytes()).decode('utf-8'),
        'dtype': str(arr.dtype),
        'shape': arr.shape
      }

      await send_message(
        'port_value',
        value_b64,
        message['id']
      )

    # create filter
    elif message['header'] == 'create_filter':
      f = filter_list[message['payload']]()

    elif message['header'] == 'delete_filter':
      filters = [f for f in pycinema.Core.Filter._filters if f.id in message['payload']]
      for f in filters:
        f.delete()

    # connect ports
    elif message['header'] == 'connect_ports':
      f0_id = message['payload'][0]['parent']
      f1_id = message['payload'][1]['parent']
      f0 = next(f for f in pycinema.filters.Filter._filters.values() if f.id == f0_id)
      f1 = next(f for f in pycinema.filters.Filter._filters.values() if f.id == f1_id)

      if message['payload'][0]['is_input']:
        p0 = f0.inputs.get(message['payload'][0]['name'])
      else:
        p0 = f0.outputs.get(message['payload'][0]['name'])

      if message['payload'][1]['is_input']:
        p1 = f1.inputs.get(message['payload'][1]['name'])
      else:
        p1 = f1.outputs.get(message['payload'][1]['name'])
      if p1.is_input:
        p0,p1 = p1,p0

      p0.set(p1)

    elif message['header'] == 'port_set_value':
      port = message['payload'][0]
      value = message['payload'][1]
      f = next(f for f in pycinema.filters.Filter._filters.values() if f.id == port['parent'])
      p = f.inputs.get(port['name']) if port['is_input'] else f.outputs.get(port['name'])
      p.set(value)

    # if message.header == 'create_filter':
    #   f = pycinema.filters.CinemaDatabaseReader()


# Static file handler for serving the 'dist' folder
async def static_file_handler(request):
    # Return the index.html when any URL is accessed
    return web.FileResponse(os.path.join('pycinema/web/client/dist', 'index.html'))

# WebSocket connection setup
async def start_websocket_server():
    # Start WebSocket server on localhost:8765
    return await websockets.serve(echo, "localhost", 8765)

async def send_message(header,payload,id=-1):
  msg = json.dumps({
    'id': id,
    'header': header,
    'payload': payload
  })
  msg_compressed = zlib.compress(msg.encode('utf-8'))
  msg_compressed_b64 = base64.b64encode(msg_compressed).decode('utf-8')

  to_remove = []
  for socket in websocket_list:
    try:
      await socket.send(msg_compressed_b64)
    except:
      to_remove.append(socket)

  for socket in to_remove:
    websocket_list.remove(socket)


def serialize(obj):
    """Recursively serialize an object, converting any with .toJSON() to JSON-serializable form."""
    if hasattr(obj, 'toJSON') and callable(obj.toJSON):
        return serialize(obj.toJSON())  # Recursively serialize the result of toJSON()
    elif isinstance(obj, dict):
        return {serialize(k): serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [serialize(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def make_async(event_name):
    async def listener(data):
        await send_message(event_name, serialize(data))
    return listener

async def init():
    app = web.Application()

    # Serve static files from the 'dist' folder
    # app.router.add_static('/assets', path='pycinema/web/client/dist/assets', name='assets')

    app.router.add_get('/', static_file_handler)

    websocket_server = await start_websocket_server()

    # Run the aiohttp web server on port 8000
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8000)
    await site.start()

    # p0 = pycinema.filters.PerformanceTest()
    # p1 = pycinema.filters.PerformanceTest()
    # p2 = pycinema.filters.PerformanceTest()

    cr = pycinema.filters.CinemaDatabaseReader()
    cr.inputs.path.set('/home/jones/2tb/projects/pycinema-data/sphere.cdb/');
    q = pycinema.filters.TableQuery()
    q.inputs.sql.set('Select * from input LIMIT 5')
    q.inputs.table.set(cr.outputs.table)
    ir = pycinema.filters.ImageReader()
    ir.inputs.table.set(q.outputs.table)
    iv = pycinema.filters.ImageView()
    iv.inputs.images.set(ir.outputs.images)


    print("Serving static files and WebSocket server on ws://localhost:8765 and http://localhost:8000")

    commands = [
      'filter_created',
      'filter_deleted',
      'filter_status',
      'value_set',
      'connection_added',
      'connection_removed',
      'update_status'
    ]

    for c in commands:
      pycinema.Filter.on(c, make_async(c))

    await websocket_server.wait_closed()

# Run the app
if __name__ == "__main__":
    asyncio.run(init())
