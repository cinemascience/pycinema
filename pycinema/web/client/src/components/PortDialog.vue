<script lang="ts" setup>
import { useDialogPluginComponent } from 'quasar';
import { reactive, onMounted, ref, nextTick } from 'vue';

import WebSocketCommunicator from '../WebSocketCommunicator.js'

const canvasContainer = ref(null);

const props = defineProps({
  port: Object
})

const iProps = reactive({
  port_str: ''
});

defineEmits([
  ...useDialogPluginComponent.emits
]);

const { dialogRef, onDialogHide, onDialogOK, onDialogCancel } = useDialogPluginComponent();


function decodeNumpyPayload(array) {
  const base64Str = array.data;
  const dtype = array.dtype;
  const shape = array.shape;

  // Decode base64 to binary
  const byteArray = Uint8Array.from(atob(base64Str), c => c.charCodeAt(0));

  // Map dtype to JS TypedArray
  let typedArray;
  switch (dtype) {
    case 'float32':
      typedArray = new Float32Array(byteArray.buffer);
      break;
    case 'float64':
      typedArray = new Float64Array(byteArray.buffer);
      break;
    case 'int32':
      typedArray = new Int32Array(byteArray.buffer);
      break;
    case 'uint8':
      typedArray = new Uint8Array(byteArray.buffer);
      break;
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }

  // Optional: reshape if needed (only 1D is natively supported in JS)
  return {
    data: typedArray,
    shape,
    dtype
  };
}

const temp = async (canvas,index)=>{
  const msg = await WebSocketCommunicator.sendMessageAsync(
    'get_port_value',
    [
      props.port.parent,
      props.port.name,
      [index,'channels','rgba']
    ]
  );
  const rgba = decodeNumpyPayload(msg.payload);

  const ctx = canvas.getContext('2d');
  const rows = rgba.shape[0];
  const cols = rgba.shape[1];

  canvas.width = cols;
  canvas.height = rows;

  const imageData = ctx.createImageData(cols, rows);
  imageData.data.set(rgba.data);
  ctx.putImageData(imageData, 0, 0);
};

const init = async ()=>{

  const values = Array.isArray(props.port.value) ? props.port.value : [props.port.value];

  console.log(values)
  for(let i=0; i<values.length; i++){
    const value = values[i];
    if(value.hasOwnProperty('channels')){
      const canvas = document.createElement('canvas');
      canvasContainer._value.appendChild(canvas);

      temp(canvas,i);
    }
  }


  // const msg = await WebSocketCommunicator.sendMessageAsync('get_port_value',props.port);

  // for(let image of msg.payload.value){
  //   const canvas = document.createElement('canvas');
  //   canvasContainer._value.appendChild(canvas);

  //   const pixelData = image.channels.rgba;

  //   const ctx = canvas.getContext('2d');
  //   const rows = pixelData.length;
  //   const cols = pixelData[0].length;

  //   canvas.width = cols;
  //   canvas.height = rows;

  //   const imageData = ctx.createImageData(cols, rows);

  //   let pixelArray = [];
  //   for (let row = 0; row < rows; row++) {
  //     for (let col = 0; col < cols; col++) {
  //       const rgba = pixelData[row][col];
  //       pixelArray.push(...rgba);
  //     }
  //   }
  //   imageData.data.set(pixelArray);
  //   ctx.putImageData(imageData, 0, 0);
  // }

  // const values = Array.isArray(props.port.value) ? props.port.value : [props.port.value];

  // for(let image of values){
  //   if(!image.hasOwnProperty('channels')) continue;

  //   const msg = await WebSocketCommunicator.sendMessageAsync('get_port_value',props.port);
  //   const

  //   const canvas = document.createElement('canvas');
  //   canvasContainer._value.appendChild(canvas);

  //   const pixelData = image.channels.rgba;

  //   const ctx = canvas.getContext('2d');
  //   const rows = pixelData.length;
  //   const cols = pixelData[0].length;

  //   canvas.width = cols;
  //   canvas.height = rows;

  //   const imageData = ctx.createImageData(cols, rows);

  //   let pixelArray = [];
  //   for (let row = 0; row < rows; row++) {
  //     for (let col = 0; col < cols; col++) {
  //       const rgba = pixelData[row][col];
  //       pixelArray.push(...rgba);
  //     }
  //   }
  //   imageData.data.set(pixelArray);
  //   ctx.putImageData(imageData, 0, 0);
  // }
};

onMounted(async ()=>{
  setTimeout(init, 100)
});

</script>

<template>
  <q-dialog ref="dialogRef" @hide="onDialogHide">
    <q-card class="q-dialog-plugin" style="min-width:45em;">
      <q-card-section>
        <q-input filled readonly :label='`${props.port.parent}.${props.port.is_input?"inputs":"outputs"}.${props.port.name}`' v-model='props.port.value_str'/>
      </q-card-section>

      <q-card-section>
        <div ref='canvasContainer'></div>
      </q-card-section>
    </q-card>

  </q-dialog>
</template>
