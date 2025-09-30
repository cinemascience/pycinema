<script setup>
import { ref, onMounted, onUnmounted, reactive, watch } from 'vue';

import Scene from './Scene.js'
import WebSocketCommunicator from '../../WebSocketCommunicator.js'
import App from '../../App.js'
import { useQuasar } from 'quasar'
const $q = useQuasar();

const svg_canvas = ref(null);
const filter_browser = ref(null);
const node_editor_container = ref(null);

const init = async ()=>{
  App.props.scene = new Scene( svg_canvas.value, $q );
  node_editor_container.value.focus();
}

const openFilterBrowserDialog = ()=>{
  filter_browser._value.showPopup();
}

const keypress = e=>{
  if(e.code==='Delete'){
    const nodes = App.props.scene.getSelectedNodes();
    App.requestDeleteFilter(nodes.map(n=>n.filter.id));
  }
}

onMounted(init);
// onUnmounted();

</script>

<template>
  <div class='node_editor_container' ref='node_editor_container' @keypress='keypress' tabindex="0">
    <div style='position:absolute;top:20px;left:20px'>
      <q-btn v-show='!App.props.server_busy' size='lg' icon='sym_o_menu' dense round color="primary" @click='openFilterBrowserDialog'/>
      <q-btn v-show='App.props.server_busy'  size='lg' icon='hourglass_bottom' class='rotating' dense round color="primary" />

      <q-select
        ref='filter_browser'
        use-input
        v-model="App.props.filter_browser.selected"
        input-debounce="0"
        :options="App.props.filter_browser.list_"
        @filter="App.props.filter_browser.search"
        @update:model-value='App.requestCreateFilter'
        style="width: 250px;display:none"
        behavior="dialog"
        label-color='white'
        dense
        options-dense
      >
      </q-select>
    </div>

    <svg class='node_editor_canvas' ref='svg_canvas' style="width:100%;height:100%">
      <defs>
        <pattern id="inner-grid" width="10" height="10" patternUnits="userSpaceOnUse">
          <rect width="100%" height="100%" fill="none" stroke="#444" stroke-width="0.5" />
        </pattern>
        <pattern id="grid" width="100" height="100" patternUnits="userSpaceOnUse">
          <rect width="100%" height="100%" fill="url(#inner-grid)" stroke="#444" stroke-width="1.5" />
        </pattern>
      </defs>
    </svg>

  </div>
</template>

<style>
.node_editor_container {
  display: block;
  position: absolute;
  left:0;
  right:0;
  top:0;
  bottom:0;
  overflow: hidden;
  /*background-color: #000;*/
}

.node_content {
  background-color: #444;
  /*background-color: rgba(50,50,50,0.8);*/
  border-radius: 0.75em;
  display: inline-block;
  overflow: hidden;
  /*resize: both;*/
  /*padding: 1em;*/
  padding: 0 0.5em 0.5em 0.5em;
}

.node_content h1 {
  font-size: 1em;
  padding: 0.5em 0 0.3em 0;
  margin: 0;
  font-weight:bold;
  text-align: center;
  cursor:pointer;
}

.status0 {
  background-color: #aaa;
}
.status1 {
  background-color: #aaa;
  background-image: linear-gradient(45deg, rgba(0, 0, 0, 0.25) 25%, transparent 25%, transparent 50%, rgba(0, 0, 0, 0.25) 50%, rgba(0, 0, 0, 0.25) 75%, transparent 75%, transparent);
  animation: barStripe 0.5s linear infinite;
}
.status2 {
  background-color: limegreen;
}

.status3 {
  background-color: #c00;
}

.node_status_line {
  height: 0.4em;
  margin: 0.1em -1em 1em -1em;
  box-sizing: border-box;
  background-size: 1em 1em;
}

@keyframes barStripe {
  0% {
    background-position: 1em 0;
  }
  100% {
    background-position: 0 0;
  }
}

.selected {
  background-color: #2e5984;
}

.input_port,.output_port  {
  /*border:0.01em solid #f00;*/
  border-radius: 0.2em;
  background-color: #333;
  margin: 0.2em 0;
  display: flex;
  align-items: center;

  box-sizing: border-box;
    line-height: 1;
  transform-origin: center;
}

@keyframes pop {
0% {
    background-color: #333; /* Color A */
  }
  50% {
    background-color: #888; /* Color B */
  }
  100% {
    background-color: #333; /* Color A */
  }
}

.pop {
  animation: pop 0.5s ease;
}

.input_port input:read-only, .output_port input:read-only {
  color:#888;
}

.node_content label {
  /*border:0.01em solid #f00;*/
  padding: 0.2em;
  font-weight: bold;
  flex-shrink: 0;
}

.node_content input {
  border:0;
  background: transparent;
  outline:none;
  width: 100%;
  flex-grow: 1;
}
.node_content input:focus {
  outline:none!important;
}

.input_port_disc, .output_port_disc {
  cursor: pointer;
}

.node_content label {
  cursor: pointer;
}

</style>
