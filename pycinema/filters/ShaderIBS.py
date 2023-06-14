from .Shader import *

class ShaderIBS(Shader):
    def __init__(self):
        super().__init__(
          inputs={
            'images': [],
            'radius': 0.03,
            'samples': 32,
            'diff': 0.5,
            'silhouette': 0.01,
            'ambient': 0.2,
            'luminance': 0.2
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

uniform float silhouette;
uniform float ambient;
uniform float luminance;

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

float computeSSAO(vec2 coord){
    float depth = readDepth( coord );

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
        occlusion += calcAO( depth, pw * aspect, ph, coord );
        z = z - dz;
        l = l + DL;
    }

    return depth>0.99 ? 1.0 : 1.-occlusion/samplesF;
}

vec3 computeIBS(vec2 coord){
  vec4 rgba = texture(rgbaTex,coord);

  float ao = computeSSAO(coord);
  float depth = readDepth(coord);

  // Compute Luminance
  vec3 lumcoeff = vec3( 0.299, 0.587, 0.114 );
  vec3 luminanceF = vec3( dot( rgba.rgb, lumcoeff ) );

  // Silhouette Effect
  vec2 pixelSize = 1./resolution;
  vec3 eps = 2.0*vec3( pixelSize.x, pixelSize.y, 0 );
  float depthN = readDepth(coord + eps.zy);
  float depthE = readDepth(coord + eps.xz);
  float depthS = readDepth(coord - eps.zy);
  float depthW = readDepth(coord - eps.xz);

  float dxdz = abs(depthE-depthW);
  float dydz = abs(depthN-depthS);
  // float dxdz = dFdx(depth);
  // float dydz = dFdy(depth);

  vec3 n = normalize( vec3(dxdz, dydz, 1./silhouette) );
  vec3 lightPos = vec3(0,0,1);
  float lightInt = 1.0*dot(n,normalize(lightPos));

  vec3 outputColor = vec3( rgba.rgb * mix( vec3(ao), vec3(1.0), luminanceF * luminance ) );

  return outputColor*ambient + outputColor*lightInt;
}

void main(){

    vec2 pixelSize = 1./resolution;
    // vec3 eps = 0.5*0.5*vec3( pixelSize.x, pixelSize.y, 0 );

    float dx = pixelSize.x/8.0;
    float dy = pixelSize.y/8.0;

    // |- - x -|
    // |x - - -|
    // |- - - x|
    // |- x - -|

    vec3 c0 = computeIBS(uv + vec2(     dx, 3.0*dy) );
    vec3 c1 = computeIBS(uv + vec2(-3.0*dx,     dy) );
    vec3 c2 = computeIBS(uv + vec2( 3.0*dx,    -dy) );
    vec3 c3 = computeIBS(uv + vec2(    -dx,-3.0*dy) );

    vec3 c = (c0+c1+c2+c3)/4.0;

    //color = vec4(c, rgba.a);
    color = vec4(c, 1.0);
}

"""

    def render(self,image):
        # update texture
        self.rgbaTex.write(image.channels['rgba'].tobytes())
        self.depthTex.write(image.channels['depth'].tobytes())

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

        # init framebuffer
        self.initFramebuffer(res)

        # set uniforms
        self.program['resolution'].value = res
        self.program['radius'].value = float(self.inputs.radius.get())
        self.program['samples'].value = int(self.inputs.samples.get())
        self.program['diff_area'].value = float(self.inputs.diff.get())
        self.program['silhouette'].value = float(self.inputs.silhouette.get()*500)
        self.program['ambient'].value = float(self.inputs.ambient.get())
        self.program['luminance'].value = float(self.inputs.luminance.get())

        # create textures
        self.rgbaTex = self.createTexture(0,res,shape[2],dtype='f1')
        self.depthTex = self.createTexture(1,res,1,dtype='f4')

        for image in images:
            results.append( self.render(image) )

        self.rgbaTex.release()
        self.depthTex.release()

        self.outputs.images.set(results)

        return 1
