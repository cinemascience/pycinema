import Base from './Base.js';
import WebSocketCommunicator from '../../WebSocketCommunicator.js';
import App from '../../App.js';

const stopEvent = e=>{
  e.preventDefault();
  e.stopPropagation();
};

const port_interaction = [null,null];

function createElementFromHTML(htmlString) {
  var div = document.createElement('div');
  div.innerHTML = htmlString.trim();
  return div.firstChild;
}

class Node extends Base {
  constructor(filter,svg,node_layer){
    super();

    this.svg = svg;

    this.filter = filter;
    this.root = node_layer
      .append("g")
      .attr('transform','translate(0,0)');

    this.xhtml = this.root
      .append("foreignObject")
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', 1000)
      .attr('height', 200)
    ;

    this.div = this.xhtml
      .append("xhtml:div")
      .attr('class','node_content')
    ;

    // const table = createElementFromHTML('<table><tr><td></td><td></td></tr></table>');
    // this.div.node().appendChild(table);
    // const row = table.getElementsByTagName('tr')[0];
    // row.children[0].appendChild(createElementFromHTML(`<i></i>`));
    // row.children[1].appendChild(createElementFromHTML(`<h1>${filter.id}</h1>`));

    this.div.node().appendChild(createElementFromHTML(`<h1>${filter.id}</h1>`));
    this.div.node().appendChild(createElementFromHTML(`<div class='node_status_line'></div>`));
    for(let _ of ['output','input'])
      for(let p of this.filter[_+'s']){
        const label = document.createElement('label');

        label.addEventListener('click',()=>this.trigger('port_clicked',p));

        label.innerHTML = p.name;
        const input = document.createElement('input');
        input.setAttribute('type','text');
        input.setAttribute('value',p.value);
        input.style['text-align'] = _==='input'?'right':'left';
        p.input = input;
        p.input.mute = false;
        if(_==='input')
          input.addEventListener('change', ()=>{
            !p.input.mute && WebSocketCommunicator.sendMessage( 'port_set_value', [p, p.type.includes('int') ? parseFloat(p.input.value) : p.input.value] );
          });
        else
          input.setAttribute('readonly','true');

        const div = document.createElement('div');
        div.className = _+'_port';
        div.appendChild(_==='input'?label:input);
        div.appendChild(_==='input'?input:label);
        this.div.node().appendChild(div);
      }

    // make node dragable
    const root_dom = this.svg.root.node();
    {
      const svg_js = this.svg.node();

      const node_head = this.div.select('h1');

      let e0 = null;
      let p0 = null;
      const drag_s = e=>{
        this.trigger('clicked', this);
        // console.log('s',e);
        stopEvent(e);
        e0 = this.clientToSVG(e.clientX,e.clientY,root_dom);
        p0 = this.root.attr('transform').split('(')[1].slice(0,-1).split(',').map(x=>parseFloat(x));
        svg_js.addEventListener('mouseleave', drag_e);
        svg_js.addEventListener('mouseup', drag_e);
        svg_js.addEventListener('mousemove', drag_i);
      };
      const drag_i = e=>{
        // console.log('i',e);
        stopEvent(e);
        const e1 = this.clientToSVG(e.clientX,e.clientY,root_dom);
        this.moveTo(
          p0[0]-(e0.x-e1.x),
          p0[1]-(e0.y-e1.y)
        );
      };
      const drag_e = e=>{
        // console.log('e',e);
        stopEvent(e);
        svg_js.removeEventListener('mouseleave', drag_e);
        svg_js.removeEventListener('mouseup', drag_e);
        svg_js.removeEventListener('mousemove', drag_i);
      };

      node_head.on('mousedown', drag_s);
    }

    // resize node container
    this.resize();

    // add port discs
    {
      const bb_n = this.div.node().getBoundingClientRect();

      // inputs
      for(let _ of ['input','output',]){
        const port_inputs = this.xhtml.selectAll(`.${_}_port`)._groups[0];
        this[`${_}PortDiscs`] = [];
        const ports = filter[`${_}s`];

        for(let i in ports){
          const port = ports[i];
          const port_input = port_inputs[i];
          const bb_p = port_input.getBoundingClientRect();

          const pn = this.clientToSVG(bb_n.x+(_==='input'?0:bb_n.width),bb_n.y,root_dom);
          const p0 = this.clientToSVG(bb_p.x,bb_p.y,root_dom);
          const p1 = this.clientToSVG(bb_p.x,bb_p.y+bb_p.height,root_dom);

          const disc = this.root
            .append('circle')
            .attr('class',`${_}_port_disc`)
            .property('filter',filter)
            .property('port',port)
            .attr('r',"5")
            .attr('fill','white')
            .attr('cx',pn.x)
            .attr('cy',p0.y-(p0.y-p1.y)/2)
          ;
          this[`${_}PortDiscs`].push(disc);
        }
      }
    }

    // port interactions
    {

      const port_discs = this.root.selectAll('circle');
      port_discs.on('mousedown', e=>{
        stopEvent(e);
        port_interaction[0] = e.target.port;
      });
      port_discs.on('mouseup', e=>{
        stopEvent(e);
        port_interaction[1] = e.target.port;
        if(port_interaction[0].parent!==port_interaction[1].parent)
          App.requestAddConnection(port_interaction);
      });
    }
  }

  delete(){
    this.root.remove();
  }

  clientToSVG(clientX, clientY, parent){
    const svg_js = this.svg.node();
    var p = svg_js.createSVGPoint();
    p.x = clientX;
    p.y = clientY;
    return p.matrixTransform(parent.getScreenCTM().inverse());
  }

  setStatus(status,error){
    this.filter.error = error;
    const node_status_line = this.div.node().getElementsByClassName('node_status_line')[0];
    const classes = node_status_line.classList;
    for(let i=0;i<4;i++)
      status===i
        ? classes.add('status'+status)
        : classes.remove('status'+i)
      ;

  }

  getPos(){
    return this.root.attr('transform').split('translate(')[1].split(',').slice(0,2).map(parseFloat);
  }

  moveTo(x,y){
    this.root.attr('transform',`translate(${x},${y})`);
    this.trigger('moved');
  }

  resize(){
    const root = this.svg.root;
    const scale = parseFloat(root.attr('transform').split('scale(').pop().split(')')[0]);
    const bb = this.div.node().getBoundingClientRect();
    this.xhtml
      .attr('width', bb.width/scale)
      .attr('height', bb.height/scale);
  }
}

export default Node;
