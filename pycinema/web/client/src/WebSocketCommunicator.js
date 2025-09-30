const WebSocketCommunicator = {
  listeners: new Map(),
  promises: new Map(),
  socket: new WebSocket('ws://localhost:8765'),

  sendMessage: (header,payload,id)=>{
    WebSocketCommunicator.socket.send(JSON.stringify({
      id: id || Math.random(),
      header: header,
      payload: payload
    }));
  },

  sendMessageAsync: async (header,payload)=>{
    const id = Math.random();
    const promise = new Promise((resolve,reject)=>{
      WebSocketCommunicator.promises.set(id,resolve);
    });
    WebSocketCommunicator.sendMessage(header,payload,id);
    return promise;
  },

  trigger: (event,data)=>{
    if(!WebSocketCommunicator.listeners.has(event)) return;
    for(let l of WebSocketCommunicator.listeners.get(event))
      l(data);
  },

  on: (event, callback)=>{
    if(!WebSocketCommunicator.listeners.has(event))
      WebSocketCommunicator.listeners.set(event,[callback]);
    else
      WebSocketCommunicator.listeners.get(event).push(callback);
  },
};

import { inflate } from 'pako';

function decompressWithPako(base64String) {
  const compressed = Uint8Array.from(atob(base64String), c => c.charCodeAt(0));
  const decompressed = inflate(compressed);
  return new TextDecoder().decode(decompressed);
}

WebSocketCommunicator.socket.onmessage = async event=>{
  const msg_raw = decompressWithPako(event.data);
  const msg = JSON.parse(msg_raw);
  // console.log(msg)
  // const msg = JSON.parse(event.data);
  if(WebSocketCommunicator.promises.has(msg.id)){
    const promise = WebSocketCommunicator.promises.get(msg.id);
    WebSocketCommunicator.promises.delete(msg.id);
    return promise(msg);
  }
  WebSocketCommunicator.trigger('message',msg);
};
WebSocketCommunicator.socket.onopen = event=>{
  WebSocketCommunicator.trigger('open',event);
};
WebSocketCommunicator.socket.onclose = event=>{
  WebSocketCommunicator.trigger('close',event);
};
WebSocketCommunicator.socket.onerror = event=>{
  console.error("WebSocket Error:", error);
  WebSocketCommunicator.trigger('error',event);
};

export default WebSocketCommunicator;
