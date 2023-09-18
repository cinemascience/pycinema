from .Shader import *

class ShaderPhong(Shader):
    def __init__(self):
        super().__init__(
          inputs={
            'images': [],
            'ambient': 0.7,
            'diffuse': 0.7,
            'specular': 1.0,
            'exponent': 16.0
          },
          outputs={
            'images': []
          },
          textures=['rgbaTex','depthTex']
        )

    def getFragmentShaderCode(self):
        return """
#version 330

in vec2 uv;
out vec4 outcolor;

uniform sampler2D rgbaTex;
uniform sampler2D depthTex;

uniform vec2 uResolution;

uniform float uAmbient;
uniform float uDiffuse;
uniform float uSpecular;
uniform float uExponent;

float readDepth( const in vec2 coord ){
  return texture( depthTex, coord ).r;
}

vec3 computeNormal(in vec2 uv, in float depth){
  vec2 pixelSize = 1./uResolution;
  vec3 eps = vec3( pixelSize.x, pixelSize.y, 0 );
  float depthN = readDepth(uv.xy + eps.zy);
  float depthS = readDepth(uv.xy - eps.zy);
  float depthE = readDepth(uv.xy + eps.xz);
  float depthW = readDepth(uv.xy - eps.xz);
  // vec3 dx = vec3(2.0*eps.xz,depthE-depthW);
  // vec3 dy = vec3(2.0*eps.zy,depthN-depthS);
  vec3 dx = vec3(eps.xz, abs(depth-depthW) < abs(depth-depthE)
    ? depthW-depth
    : depth-depthE
  );
  vec3 dy = vec3(eps.zy, abs(depth-depthN) < abs(depth-depthS)
    ? depth-depthN
    : depthS-depth
  );
  return normalize(cross(dx, dy));
}

vec4 compute(const in vec2 sampleUV, const in vec2 pixelUV){
    vec4 pixelAlbedoRGBA = texture2D( rgbaTex, pixelUV );
    vec4 sampleAlbedoRGBA = texture2D( rgbaTex, sampleUV );
    vec3 albedo = sampleAlbedoRGBA.rgb;
    float alpha = sampleAlbedoRGBA.a;
    float depth = readDepth(sampleUV);
    vec3 normal = computeNormal(sampleUV, depth);
    vec3 lightDir = normalize(vec3(1,1,1));
    vec3 viewDir = vec3(0,0,1);
    vec3 halfDir = normalize(lightDir + viewDir);
    float lambertian = max(dot(lightDir, normal), 0.0);
    float specAngle = max(dot(halfDir, normal), 0.0);
    float specular = lambertian > 0.0 ? pow(specAngle, uExponent) : 0.0;
    vec3 diffuseColor = albedo * lambertian;
    vec3 color = albedo*uAmbient + diffuseColor*uDiffuse + specular*uSpecular;
    return alpha<1.0 || depth==1.0
      ? vec4(pixelAlbedoRGBA.rgb,floor(alpha))
      : vec4(color,1.0)
    ;
}

void main(){
  vec2 pixelSize = 1.0/uResolution;
  vec2 eps = vec2(1,-1)*0.15;
  vec4 color = (
    compute(uv+eps.xx*pixelSize, uv)
    +compute(uv+eps.xy*pixelSize, uv)
    +compute(uv+eps.yy*pixelSize, uv)
    +compute(uv+eps.yx*pixelSize, uv)
  );
  outcolor = color/4.0;
}

"""

    def render(self,image):
        # update framebuffer and textures
        self.initFramebuffer(image.resolution)
        self.updateTexture(0,image.getChannel('rgba'))
        self.updateTexture(1,image.getChannel('depth'))
        self.program['uResolution'].value = image.resolution

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
        self.program['uAmbient'].value = float(self.inputs.ambient.get())
        self.program['uDiffuse'].value = float(self.inputs.diffuse.get())
        self.program['uSpecular'].value = float(self.inputs.specular.get())
        self.program['uExponent'].value = float(self.inputs.exponent.get())

        # render images
        try:
          for image in images:
              results.append( self.render(image) )
        except:
          self.outputs.images.set(images)
          return 1

        self.outputs.images.set(results)

        return 1
