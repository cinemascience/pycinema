from .Core import *

import numpy
import moderngl

class Shader(Filter):
    def __init__(self):
        super().__init__()

        # create context
        self.ctx = moderngl.create_standalone_context(require=330)
        # self.ctx.release()

        # fullscreen quad
        self.quad = self.ctx.buffer(
            numpy.array([
                 1.0,  1.0,
                -1.0,  1.0,
                -1.0, -1.0,
                 1.0, -1.0,
                 1.0,  1.0
            ]).astype('f4').tobytes()
        )

    def init(self,textureNames):
        # program
        self.program = self.ctx.program(
            vertex_shader=self.getVertexShaderCode(),
            fragment_shader=self.getFragmentShaderCode(),
            varyings=["uv"]
        )
        for i, name in enumerate(textureNames):
            self.program[name] = i

        self.vao = self.ctx.simple_vertex_array(self.program, self.quad, 'position')

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
