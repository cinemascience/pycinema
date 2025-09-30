<script lang="ts" setup>
import { useDialogPluginComponent } from 'quasar';
import { reactive, onMounted, ref, nextTick } from 'vue';

const props = defineProps({
  title: String,
  msg: String
})

defineEmits([
  ...useDialogPluginComponent.emits
]);

const { dialogRef, onDialogHide, onDialogOK, onDialogCancel } = useDialogPluginComponent();

</script>

<template>
  <q-dialog ref="dialogRef" @hide="onDialogHide">
    <q-card flat class="q-dialog-plugin" style="width:auto;max-width:50vw;min-width:20em;">
      <q-card-section>
        <div class="text-h6">{{props.title}}</div>
      </q-card-section>

      <q-card-section class="row no-wrap items-center" style="padding:0 2em 0 0.5em;">
        <q-icon class='col-grow' name="warning" color="grey-7" size="2em" style="padding:0 0.25em"/>
        <div class="col" v-html='props.msg' style="white-space:nowrap"></div>
      </q-card-section>

      <q-card-actions align="right" style="margin:0 1.5em 1.5em;">
        <q-btn color='primary' label='OK' @click='onDialogOK' class='text-weight-bold'/>
      </q-card-actions>
    </q-card>
  </q-dialog>
</template>
