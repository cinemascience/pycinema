from pycinema import Filter

import numpy
import moderngl

class Shader(Filter):

    ctx = None
    quad = None
    fbo = None

    def __init__(self, inputs={}, outputs={}, textures=[], varyings=['uv']):
        # program
        self.program = Shader.ctx.program(
            vertex_shader=self.getVertexShaderCode(),
            fragment_shader=self.getFragmentShaderCode(),
            varyings=varyings
        )

        # textures
        for i, name in enumerate(textures):
            self.program[name] = i

        # Geometry
        self.vao = Shader.ctx.simple_vertex_array(self.program, Shader.quad, 'position')

        super().__init__(inputs, outputs)

    def initFramebuffer(self,res,components=[1],dtypes=['f1']):
        if Shader.fbo==None or Shader.fbo.size!=res:
            if Shader.fbo!=None:
                Shader.fbo.release()
            if len(components)==1:
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

    def createTexture(self,location,res,components,dtype='f1'):
        tex = self.ctx.texture(res, components, dtype=dtype, alignment=1)
        tex.repeat_x = False
        tex.repeat_y = False
        tex.use(location=location)
        return tex

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
    print("WARNING: Unable to setup OpenGL context.")
