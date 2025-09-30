class Edge {
  constructor([p0,p1],[n0,n1],svg,edge_layer){
    this.svg = svg;
    this.root = svg.root._groups[0][0];
    this.p0 = p0;
    this.p1 = p1;

    const p0Index = n0.filter.outputs.map(x=>x.name).indexOf(p0.name);
    const p1Index = n1.filter.inputs.map(x=>x.name).indexOf(p1.name);

    this.n0_port = n0.outputPortDiscs[p0Index];
    this.n1_port = n1.inputPortDiscs[p1Index];

    this.path = edge_layer.append('path')
      .attr('d',``)
      .attr('stroke','#1976d2')
      .attr('stroke-width','5')
      .attr('fill','none')
    ;

    n0.on('moved', ()=>this.update_pos());
    n1.on('moved', ()=>this.update_pos());

    this.update_pos();
  }

  clientToSVG(clientX, clientY, parent){
    const svg_js = this.svg._groups[0][0];
    var p = svg_js.createSVGPoint();
    p.x = clientX;
    p.y = clientY;
    return p.matrixTransform(parent.getScreenCTM().inverse());
  }

  update_pos(){
    const n0_port_bb = this.n0_port._groups[0][0].getBoundingClientRect();
    const n1_port_bb = this.n1_port._groups[0][0].getBoundingClientRect();
    const pos0 = this.clientToSVG(n0_port_bb.x+n0_port_bb.width/2,n0_port_bb.y+n0_port_bb.height/2,this.root);
    const pos1 = this.clientToSVG(n1_port_bb.x+n1_port_bb.width/2,n1_port_bb.y+n1_port_bb.height/2,this.root);

    let dx = Math.abs(pos0.x - pos1.x);
    if(pos0.x<pos1.x)
      dx *= 0.5;

    this.path
      .attr('d',`
        M ${pos0.x},${pos0.y}
        C
        ${pos0.x+dx},${pos0.y}
        ${pos1.x-dx},${pos1.y}
        ${pos1.x},${pos1.y}
      `)
    ;
  }
}

export default Edge;
