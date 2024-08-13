from pycinema import Filter

import numpy
import moderngl
import re
import logging as log

class Shader(Filter):

    ctx = None
    quad = None
    fbo = None

    def __init__(self, inputs={}, outputs={}, textures=[], varyings=['uv'], quad=True):

        # program
        self.program = Shader.ctx.program(
            vertex_shader=self.getVertexShaderCode(),
            fragment_shader=self.getFragmentShaderCode(),
            varyings=varyings
        )

        # textures
        self.textures = {}
        for i, name in enumerate(textures):
            self.program[name] = i

        # Geometry
        if quad:
            self.vao = Shader.ctx.simple_vertex_array(self.program, Shader.quad, 'position')

        Filter.on('filter_deleted', lambda f: 0 if f!=self else self.releaseTextures())

        super().__init__(inputs, outputs)

    def initFramebuffer(self,res,components=[1],dtypes=['f1']):
        if Shader.fbo==None or Shader.fbo.size!=res:
            if Shader.fbo!=None:
                Shader.fbo.release()
            if len(components)==1 and dtypes[0]=='f1':
                Shader.fbo = Shader.ctx.simple_framebuffer(res)
            else:
                attachments = []
                for a in range(len(components)):
                    attachments.append( Shader.ctx.renderbuffer(size=res, components=components[a], dtype=dtypes[a]) )
                Shader.fbo = Shader.ctx.framebuffer( color_attachments=attachments )
            Shader.fbo.use()
        return Shader.fbo

    def readFramebuffer(self,attachment=0,components=4,dtype=numpy.uint8):
        b = self.fbo.read(attachment=attachment,components=components, dtype = dtype==numpy.uint8 and 'f1' or 'f4' )
        fa = numpy.frombuffer(b, dtype=dtype)
        a = fa.view()
        if components > 1:
          a.shape = (self.fbo.size[1],self.fbo.size[0],components)
        else:
          a.shape = (self.fbo.size[1],self.fbo.size[0])
        return a

    def getVertexShaderCode(self):
        return """
#version 330

in vec2 position;
out vec2 uv;

void main(){
    uv = position/2.0+0.5;
    gl_Position = vec4(position,0,1);
}
"""

    def getFragmentShaderCode(self):
        # needs to be overriden
        return ""

    def updateTexture(self,location,data,skip_write_if_exists=False):
        if location not in self.textures or self.textures[location].shape!=data.shape:
            dtype_map = {
              '|u1': 'f1',
              '<f4': 'f4'
            }
            tex = self.ctx.texture(data.shape[:2][::-1], 1 if len(data.shape)<3 else data.shape[2], dtype=dtype_map[data.dtype.str], alignment=1)
            tex.repeat_x = False
            tex.repeat_y = False
            tex.use(location=location)
            tex.write(data.tobytes())
            tex.shape = data.shape
            tex.resolution = data.shape[:2][::-1]
            self.textures[location] = tex
        else:
            tex = self.textures[location]
            tex.use(location=location)
            if not skip_write_if_exists:
                tex.write(data.tobytes())

    def releaseTextures(self):
        for t in [t for t in self.textures.keys()]:
            self.textures[t].release()
            del self.textures[t]

try:
    Shader.ctx = moderngl.create_standalone_context(require=330)
    Shader.quad = Shader.ctx.buffer(
        numpy.array([
             1.0,  1.0,
            -1.0,  1.0,
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0
        ]).astype('f4').tobytes()
    )
except:
    log.warning("Unable to setup OpenGL context.")
