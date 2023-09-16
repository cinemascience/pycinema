from .Shader import *

class ShaderSSAO(Shader):
    def __init__(self):
        super().__init__(
          inputs={
            'images': [],
            'radius': 0.03,
            'samples': 32,
            'diff': 0.5
          },
          outputs={
            'images': []
          },
          textures=['rgbaTex','depthTex']
        )

    def getFragmentShaderCode(self):
        return """
#version 330

uniform sampler2D rgbaTex;
uniform sampler2D depthTex;
uniform float radius;
uniform float diff_area;
uniform int samples;
uniform vec2 resolution;

in vec2 uv;
out vec4 color;

#define DL 2.399963229728653  // PI * ( 3.0 - sqrt( 5.0 ) )
#define EULER 2.718281828459045

float readDepth(vec2 coord){
    //float d = texture(depthTex,coord).r;
    //return isnan(d) ? 1.0 : d;
    return texture(depthTex,coord).r;
}

const float gDisplace = 0.5;  // gauss bell center
float compareDepths( const in float depth1, const in float depth2, inout int far ) {
    float garea = 16.0;        // gauss bell width
    float diff = ( depth1 - depth2 ) * 100.0;  // depth difference (0-100)

    // reduce left bell width to avoid self-shadowing
    if(diff<gDisplace){
        garea = diff_area;
    } else {
        far = 1;
    }

    float dd = diff - gDisplace;
    return pow( EULER, -2.0 * ( dd * dd ) / ( garea * garea ) );
}

float calcAO( float depth, float dw, float dh, vec2 uv ) {
    vec2 vv = vec2( dw, dh );
    vec2 coord1 = uv + vv;
    vec2 coord2 = uv - vv;
    float temp1 = 0.0;
    float temp2 = 0.0;
    int far = 0;

    temp1 = compareDepths( depth, readDepth( coord1 ), far );
    if ( far > 0 ) {
        temp2 = compareDepths( readDepth( coord2 ), depth, far );
        temp1 += ( 1.0 - temp1 ) * temp2;
    }
    return temp1;
}

void main(){
    float depth = readDepth( uv );

    float samplesF = samples;
    float occlusion = 0.0;

    float dz = 1.0 / samplesF;
    float l = 0.0;
    float z = 1.0 - dz / 2.0;

    float aspect = resolution.y/resolution.x;

    for(int i=0; i<samples; i++){
        float r = sqrt( 1.0 - z ) * radius;
        float pw = cos( l ) * r;
        float ph = sin( l ) * r;
        occlusion += calcAO( depth, pw * aspect, ph, uv );
        z = z - dz;
        l = l + DL;
    }

    float ao = depth>0.99 ? 1.0 : 1.-occlusion/samplesF;

    vec4 rgba = texture(rgbaTex,uv);
    color = vec4(mix(vec3(0),rgba.rgb,ao+0.2),rgba.a);
}

"""

    def render(self,image):
        # update framebuffer and textures
        self.initFramebuffer(image.resolution)
        self.updateTexture(0,image.getChannel('rgba'))
        self.updateTexture(1,image.getChannel('depth'))
        self.program['resolution'].value = image.resolution

        # render
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # read framebuffer
        outImage = image.copy()
        outImage.channels['rgba'] = self.readFramebuffer()

        return outImage

    def _update(self):
        results = []
        images = self.inputs.images.get()

        # set uniforms
        self.program['radius'].value = float(self.inputs.radius.get())
        self.program['samples'].value = int(self.inputs.samples.get())
        self.program['diff_area'].value = float(self.inputs.diff.get())

        # render images
        try:
          for image in images:
              results.append( self.render(image) )
        except:
          self.outputs.images.set(images)
          return 1

        self.outputs.images.set(results)

        return 1
