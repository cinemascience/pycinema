import { createApp } from 'vue'
import { Quasar,Dialog } from 'quasar'

// Import icon libraries
import '@quasar/extras/material-icons/material-icons.css'
import '@quasar/extras/material-icons-outlined/material-icons-outlined.css'
import '@quasar/extras/material-icons-round/material-icons-round.css'
import '@quasar/extras/material-symbols-outlined/material-symbols-outlined.css'

// Import Quasar css
import 'quasar/dist/quasar.css'

import './style.css'

import App from './App.vue'

const myApp = createApp(App)

myApp.use(Quasar, {
  plugins: {Dialog}, // import Quasar plugins and add here
  config:{dark: true}
})

// Assumes you have a <div id="app"></div> in your index.html
myApp.mount('#app')
