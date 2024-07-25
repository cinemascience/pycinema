from .Shader import *

import pycinema

import numpy
import moderngl
import PIL

# try:

# except NameError:
#   pass

class MeshRenderer(Shader):

    def __init__(self):

        super().__init__(
          inputs={
            'data': None,
            'cameras': [],
            'resolution': (256,256),
          },
          outputs={
            'images': []
          }
        )

    def getVertexShaderCode(self):
        return """
#version 330

in vec3 position;
out vec2 uv;

void main(){
    uv = vec2(1,-1)*position.xy;
    gl_Position = vec4(position,1);
}
"""

    def getFragmentShaderCode(self):
        return """
#version 330

in vec2 uv;
layout(location=0) out float outDepth;

void main() {
    outDepth = uv.x;
}
        """

    def render(self):

        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.mesh_vao.render(moderngl.TRIANGLE_STRIP)

        # create output image
        return pycinema.Image(
            { 'depth': self.readFramebuffer(0,1,numpy.float32) },
            {}
        )

    def initScene(self):

        self.mesh = Shader.ctx.buffer(
            numpy.array([
                 1.0,  1.0, 0.0,
                -1.0,  1.0, 0.0,
                -1.0, -1.0, 0.0
            ]).astype('f4').tobytes()
        )

        self.mesh_vao = Shader.ctx.simple_vertex_array(
          self.program,
          self.mesh,
          'position'
        )

    def _update(self):

        # create framebuffer
        res = self.inputs.resolution.get()
        self.initFramebuffer(res,[1],['f4'])

        self.initScene()

        result = [self.render()]

        self.outputs.images.set(result);

        return 1;
