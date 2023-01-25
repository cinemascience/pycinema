from .Shader import *

class ShaderSSAO(Shader):
    def __init__(self):
        super().__init__()
        self.addInputPort("images", [])
        self.addInputPort("radius", 0.03)
        self.addInputPort("samples", 32)
        self.addInputPort("diff", 0.5)
        self.addOutputPort("images", [])

        self.init(['rgbaTex','depthTex'])

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
        rgba = image.channels['rgba']
        depth = image.channels['depth']

        # create texture
        self.rgbaTex.write(rgba.tobytes())
        self.depthTex.write(depth.tobytes())

        # render
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # read pixels
        rgbaBuffer = self.fbo.read(attachment=0,components=4)
        rgbaFlatArray = numpy.frombuffer(rgbaBuffer, dtype=numpy.uint8)
        rgbaArray = rgbaFlatArray.view()
        rgbaArray.shape = (self.fbo.size[1],self.fbo.size[0],4)

        outImage = image.copy()
        outImage.channels['rgba'] = rgbaArray

        return outImage

    def update(self):

        results = []

        images = self.inputs.images.get()
        if len(images)<1:
            self.outputs.images.set(results)
            return 1

        # first image
        image0 = images[0]
        if not 'depth' in image0.channels or not 'rgba' in image0.channels:
            self.outputs.images.set(images)
            return 1

        shape = image0.channels['rgba'].shape
        if len(shape)!=3:
            shape = (shape[0],shape[1],1)
        res = shape[:2][::-1]

        # set uniforms
        self.program['resolution'].value = res
        self.program['radius'].value = float(self.inputs.radius.get())
        self.program['samples'].value = int(self.inputs.samples.get())
        self.program['diff_area'].value = float(self.inputs.diff.get())

        # create framebuffer
        self.fbo = self.ctx.simple_framebuffer(res)
        self.fbo.use()

        # create textures
        self.rgbaTex = self.createTexture(0,res,shape[2],dtype='f1')
        self.depthTex = self.createTexture(1,res,1,dtype='f4')

        for image in images:
            results.append( self.render(image) )

        self.rgbaTex.release()
        self.depthTex.release()
        self.fbo.release()

        self.outputs.images.set(results)

        return 1
