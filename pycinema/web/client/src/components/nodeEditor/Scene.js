import Node from './Node.js';
import Edge from './Edge.js';
import WebSocketCommunicator from '../../WebSocketCommunicator.js';
import App from '../../App.js';
import { instance } from "@viz-js/viz";

import { watch } from "vue";

import PortDialog from '../PortDialog.vue';
import ConfirmationDialog from '../ConfirmationDialog.vue';

import * as d3 from 'd3';

const animation_data = {
  nodes: null,
  t0: null,
  duration: null
};

const animate_scene = () => {
  const dt = performance.now() - animation_data.t0;
  const l = Math.min(dt/animation_data.duration,1);
  for(let node of animation_data.nodes)
    node.moveTo(
      l*node.move_target[0] + (1-l)*node.move_origin[0],
      l*node.move_target[1] + (1-l)*node.move_origin[1]
    );
  if(dt<animation_data.duration)
    requestAnimationFrame(animate_scene);
};

class Scene {

  update(){
    // remove deleted nodes
    const filterIds = App.props.filters.map(f=>f.id);
    for(let id of [...this.nodes.values()].map(n => n.filter.id))
      !filterIds.includes(id) && this.removeNode(id);

    // add new nodes
    for(let f of App.props.filters)
      !this.nodes.has(f.id) && this.addNode(f);

    // update edges
    for(let f of App.props.filters){
      for(let i of f.inputs){
        if(i.portRef)
          this.addEdge([
            App.props.filters.filter(f=>f.id===i.portRef.parent)[0].outputs.filter(o=>o.name===i.portRef.name)[0],
            i
          ]);
      }
    }
  }

  constructor(svg_canvas,quasar){

    this.quasar = quasar;

    instance().then(viz=>{
      this.viz = viz;
    });

    // communicator
    WebSocketCommunicator.on('message', msg=>{
    //   console.log(msg)
      switch(msg.header){
    //     case 'filter_created':
    //       return this.addNode(msg.payload);
        case 'connection_added':
          return this.update();
        case 'filter_status':
          return this.setStatus(msg.payload);
        case 'value_set':
          const port = msg.payload[0];
          const node = this.nodes.get(port.parent);
          const port_ = node.filter[port.is_input?'inputs':'outputs'].find(i=>i.name===port.name);
          for(let key of Object.keys(port))
            port_[key] = port[key];

          const port_updates = [port_];

          port_.input.mute = true;
          port_.input.setAttribute('value',port_.value_str);
          for(let [_,n] of this.nodes)
            for(let p of n.filter.inputs)
              if(p.portRef && p.portRef.parent===port.parent && p.portRef.name===port.name)
                port_updates.push(p);

          for(let p of port_updates){
            p.input.parentNode.classList.add('pop');
            p.input.mute = false;
            if(p.is_input)
              if(p.portRef)
                p.input.setAttribute('readonly', true);
              else
                p.input.removeAttribute('readonly');
          }
          setTimeout(()=>{
            for(let p of port_updates)
              p.input.parentNode.classList.remove('pop');
          }, 1000);
          return;
      }
    });

    this.svg = d3.select(svg_canvas);
    this.nodes = new Map();
    this.edges = new Map();

    const patternGrid = this.svg.select('#grid');
    const patternInnerGrid = this.svg.select('#inner-grid');

    this.grid = this.svg.append('rect')
      .attr('x',0)
      .attr('y',0)
      .attr('width','100%')
      .attr('height','100%')
      .attr('fill','url(#grid)')
    ;

    this.root = this.svg.append('g');
    this.svg.root = this.root;
    this.edge_layer = this.root.append('g');
    this.node_layer = this.root.append('g');
    this.foreground_layer = this.root.append('g');


    this.grid.on('click', ()=>this.selectNode());

    const transformed = ({transform}) => {
      const transform10 = parseInt(transform.k * 10);
      const transform100 = transform10 * 10;

      // Don't move the grid itself, simply change the pattern.
      patternGrid
        .attr('x', parseInt(transform.x % transform100))
        .attr('y', parseInt(transform.y % transform100))
        .attr('width', transform100)
        .attr('height', transform100);
      patternInnerGrid
        .attr('width', transform10)
        .attr('height', transform10);

      // Translate and scale the canvas.
      this.root.attr('transform', transform);
    };
    const zoom = d3.zoom()
        .scaleExtent([0.25, 4])
        .on("zoom", transformed);

    this.svg.call(zoom).call(zoom.transform, d3.zoomIdentity);

    watch(()=>App.props.filters, ()=>this.update());
  }

  getSelectedNodes(){
    return [...this.nodes.values()].filter(n=>n.div.node().classList.contains('selected'));
  }

  selectNode(node){
    for(let [_,n] of this.nodes)
      n.div.node().classList.remove('selected');
    node && node.div.node().classList.add('selected');
  }

  autoConnectFilters(n0,n1){
    if(!n0 || !n1) return;
    const f0 = n0.filter;
    const f1 = n1.filter;
    for(let o of f0.outputs)
      for(let i of f1.inputs)
        if(o.name===i.name)
          WebSocketCommunicator.sendMessage(
            'connect_ports',
            [o,i]
          );
  }

  showPort(port){
    this.quasar.dialog({
      component: PortDialog,
      componentProps: {port:port}
    });
  }

  setStatus([id,status,error]){
    if(id<0){
      this.nodes.forEach(n=>n.setStatus(status,error));
    } else {
      this.nodes.get(id).setStatus(status,error);
    }
  }

  addNode(filter){
    const node = new Node(filter,this.svg,this.node_layer);
    node.moveTo(Math.random()*500,Math.random()*500);
    this.nodes.set(filter.id,node);
    this.edges.set(filter.id,[]);

    node.on('clicked',node=>this.selectNode(node));
    node.on('clicked',node=>node.filter.error &&
      this.quasar.dialog({
        component: ConfirmationDialog,
        componentProps: {title:'Error',msg:node.filter.error.replaceAll('\n','<br>').replaceAll(' ','&nbsp;')}
      })
    );
    node.on('port_clicked',port=>this.showPort(port));

    this.autoConnectFilters(this.getSelectedNodes()[0],node);

    this.selectNode(node);

    this.computeLayout();
  }

  removeNode(id){
    this.nodes.get(id).delete();
    this.nodes.delete(id);
    this.edges.delete(id);

    this.computeLayout();
  }

  addEdge(ports){
    console.log(ports)
    const edge = new Edge(
      ports,
      ports.map(p=>this.nodes.get(p.parent)),
      this.svg,
      this.edge_layer
    );
    this.edges.get(ports[0].parent).push(edge);
    this.edges.get(ports[1].parent).push(edge);

    this.computeLayout();
  }

  computeLayout(){

    let node_string = ``;
    let edge_string = ``;
    for(let [id,node] of this.nodes){
      node_string+=`${id}[shape=record,height=${node.xhtml.node().clientHeight/100},width=${node.xhtml.node().clientWidth/100},label="{ {${node.filter.outputs.map((_,i)=>`<o${i}>`).concat(node.filter.inputs.map((_,i)=>`<i${i}>`)).join('|')}} }"];\n`;

      for(let p0_idx in node.filter.inputs){
        const p0 = node.filter.inputs[p0_idx];
        if(p0.portRef){
          const p1_idx = this.nodes.get(p0.portRef.parent)
            .filter[p0.portRef.is_input?'inputs':'outputs']
            .findIndex(_=>_.name===p0.portRef.name)
          ;
          edge_string += `${p0.portRef.parent}:o${p1_idx}->${p0.parent}:i${p0_idx};\n`;
        }
      }
    }

    let dot = `digraph {rankdir = LR;\n`;
    dot+=node_string;
    dot+=edge_string;
    dot+=`}\n`;

    const layout = this.viz.renderJSON(dot);
    for(let node_layout of layout.objects){
      const node = this.nodes.get(node_layout.name);
      node.move_origin = node.getPos();
      node.move_target = node_layout.pos.split(',').map(i=>parseFloat(i)*1.7)
    }

    animation_data.t0 = performance.now();
    animation_data.duration = 200;
    animation_data.nodes = [...this.nodes.values()];
    console.log(dot)
    requestAnimationFrame(animate_scene);
  }
}
export default Scene;
