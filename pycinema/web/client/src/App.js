import {reactive} from 'vue';
import WebSocketCommunicator from './WebSocketCommunicator.js';

const App = {
  props: reactive({
    layout: {
      direction: 'horizontal',
      components: [
        ['NodeEditor',null]
      ],
    },

    scene: null,

    server_busy: false,

    filters: [],

    filter_browser: {
      list: [],
      list_: [],
      selected: null,
      search: (val, update) => {
        if (val === '')
          update(() => {
            NodeEditor.props.filter_browser.list_ = NodeEditor.props.filter_browser.list;
          });
        else
          update(() => {
            const needle = val.toLowerCase();
            NodeEditor.props.filter_browser.list_ = NodeEditor.props.filter_browser.list.filter(v => v.toLowerCase().indexOf(needle) > -1);
          });
      }
    },
  }),

  requestCreateFilter: type=>{
    NodeEditor.props.filter_browser.selected = null;
    if(!type) return;
    WebSocketCommunicator.sendMessage('create_filter',type);
  },

  requestDeleteFilter: filter_ids=>{
    WebSocketCommunicator.sendMessage('delete_filter',filter_ids);
  },
  requestAddConnection: ports=>{
    WebSocketCommunicator.sendMessage('connect_ports',ports);
  },
  requestRemoveConnection: ()=>{

  },
};


WebSocketCommunicator.on('open', async ()=>{
    console.log("Connected to WebSocket server");
    {
      const msg = await WebSocketCommunicator.sendMessageAsync('get_filter_list');
      App.props.filter_browser.list = msg.payload;
    }
    {
      const msg = await WebSocketCommunicator.sendMessageAsync('get_filters');
      App.props.filters = App.props.filters.concat(msg.payload);
    }
  });

WebSocketCommunicator.on('message', msg=>{
  console.log(msg)
  switch(msg.header){
    case 'update_status':
      return App.props.server_busy = !msg.payload;
    case 'filter_created':
      return App.props.filters = App.props.filters.concat([msg.payload]);
    case 'filter_deleted':
      return App.props.filters = App.props.filters.filter(f=>f.id!==msg.payload.id);
    // case 'connection_added':
    //   return NodeEditor.props.filters = NodeEditor.props.filters.filter(f=>f.id!==msg.payload.id);
  }
});


// setTimeout(function() {
//   App.props.layout.components.push(['ImageView','ImageView_0'])
// }, 1000);

export default App;
