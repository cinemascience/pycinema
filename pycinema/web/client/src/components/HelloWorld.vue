<script setup>
import { ref, onMounted } from 'vue'

defineProps({
  msg: String,
})

const count = ref(0)

const test = ()=>{
  sendMessage('xxxxxxxxxx')
};

const io = {};

const sendMessage = msg=>{
  io.socket.send(msg);
};

const init = async ()=>{
  // Connect to the WebSocket server
  io.socket = new WebSocket('ws://localhost:8765');

  // Set up event listener for when a message is received from the server
  io.socket.onmessage = (event) => {
    console.log('->',JSON.parse(event.data))
  };

  // Handle WebSocket connection errors
  io.socket.onerror = (error) => {
    console.error("WebSocket Error:", error);
  };

  // Handle WebSocket connection open event
  io.socket.onopen = () => {
    console.log("Connected to WebSocket server");
  };
}

onMounted(init);

</script>

<template>
  <h1>{{ msg }}</h1>

  <button @click='test'>Test</button>
</template>

<style scoped>
.read-the-docs {
  color: #888;
}
</style>
