class Base {
  constructor(){
    this.listeners = new Map();
  }

  trigger(event,data){
    if(!this.listeners.has(event)) return;
    for(let l of this.listeners.get(event))
      l(data);
  }

  on(event, callback){
    if(!this.listeners.has(event))
      this.listeners.set(event,[callback]);
    else
      this.listeners.get(event).push(callback);
  }
}

export default Base;
