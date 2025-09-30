<script setup>

import {reactive} from 'vue';
import Tile from './components/Tile.vue'
import App from './App.js'

</script>


<template>
  <div class="app-container">
    <Tile :node="App.props.layout" />
  </div>
</template>

<style>

.app-container {
  height: 100%;
  width: 100%;

  -webkit-user-select: none; /* Safari */
  -ms-user-select: none; /* IE 10 and IE 11 */
  user-select: none; /* Standard syntax */
}

.material-symbols-outlined {
  font-variation-settings:
    'FILL' 0,
    'wght' 400,
    'GRAD' 0,
    'opsz' 24
}

.rotating {
  animation: spin 2s linear infinite;
  display: inline-block; /* Ensure it rotates properly */
  transform-origin: center center;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

</style>
