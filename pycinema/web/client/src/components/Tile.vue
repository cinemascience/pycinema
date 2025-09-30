<template>
  <div
    class="tile"
    :class="direction"

  >
    <template v-for="([child,props], index) in node.components">
      <div
        class="tile-child"
        :style="{ flexBasis: iProps.sizes[index] + '%' }"

      >
        <Tile v-if="isContainer(child)" :node="child" />
        <component v-else :is="componentMap[child]" :props='props' />
      </div>

      <!-- Insert a splitter between children -->
      <div
        v-if="index < node.components.length - 1"
        class="splitter"
        :class="direction"
        @mousedown="startDragging(index, $event)"
        :key="'splitter-' + index"
      ></div>
    </template>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount,reactive } from 'vue'
import Tile from './Tile.vue'
import ComponentA from './ComponentA.vue'
import ComponentB from './ComponentB.vue'
import NodeEditor from './nodeEditor/NodeEditor.vue'
import ImageView from './filterViews/ImageView.vue'

const props = defineProps({
  node: {
    type: Object,
    required: true,
  },
})


const direction = props.node.direction
const isHorizontal = direction === 'horizontal'

const componentMap = {
  ComponentA,
  ComponentB,
  NodeEditor,
  ImageView
}

// Helper: is this a container node or a component name?
const isContainer = (child) => typeof child === 'object'

const iProps = reactive({
  sizes: props.node.components.map(() => 100 / props.node.components.length)
});

// Resizing state
let isDragging = false
let dragIndex = null
let startPos = 0
let startSizes = []

let containerSize = null;

function startDragging(index, event) {
  isDragging = true
  dragIndex = index
  startPos = isHorizontal ? event.clientX : event.clientY
  startSizes = [...iProps.sizes]
  containerSize = event.target.parentElement.getBoundingClientRect()[isHorizontal ? 'width' : 'height']
  window.addEventListener('mousemove', onDrag)
  window.addEventListener('mouseup', stopDragging)
}

function onDrag(event) {
  if (!isDragging) return
  const delta = (isHorizontal ? event.clientX : event.clientY) - startPos
  const deltaPercent = (delta / containerSize) * 100

  iProps.sizes[dragIndex] = startSizes[dragIndex]+deltaPercent
  iProps.sizes[dragIndex + 1] = startSizes[dragIndex+1]-deltaPercent

  // const total = newSizes[dragIndex] + newSizes[dragIndex + 1]
  // newSizes[dragIndex] = (newSizes[dragIndex] / total) * (startSizes[dragIndex] + startSizes[dragIndex + 1])
  // newSizes[dragIndex + 1] = 100 - newSizes[dragIndex]
}

function stopDragging() {
  isDragging = false
  window.removeEventListener('mousemove', onDrag)
  window.removeEventListener('mouseup', stopDragging)
}

onBeforeUnmount(() => {
  stopDragging()
})
</script>

<style scoped>
.tile {
  display: flex;
  width: 100%;
  height: 100%;
  overflow: hidden;
}
.horizontal {
  flex-direction: row;
}
.vertical {
  flex-direction: column;
}
.tile-child {
  overflow: hidden;
  position: relative;
}
.splitter {
  background-color: #ccc;
  z-index: 10;
}
.splitter.horizontal {
  width: 4px;
  cursor: col-resize;
}
.splitter.vertical {
  height: 4px;
  cursor: row-resize;
}
</style>
