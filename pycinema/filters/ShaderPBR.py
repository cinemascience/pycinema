from .Shader import *

class ShaderPBR(Shader):
    def __init__(self):
        super().__init__(
          inputs={
            'images': [],
            'ambient': 0.2,
            'diffuse': 1.0,
            'roughness': 0.2,
            'metallic': 0.0
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

uniform sampler2D rgbaTex; // albedo
uniform sampler2D depthTex; // depth

uniform vec2 uResolution;

uniform float uAmbient;
uniform float uDiffuse;
uniform float uRoughness;
uniform float uMetallic;

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

vec3 computeHalfVector(
  in vec3 toLight,
  in vec3 toView
){
  return normalize(toLight + toView);
}


#define PI              3.14159
#define ONE_OVER_PI     0.31831
/**
 * GGX/Trowbridge-Reitz NDF
 *
 * Calculates the specular highlighting from surface roughness.
 *
 * Roughness lies on the range [0.0, 1.0], with lower values
 * producing a smoother, "glossier", surface. Higher values
 * produce a rougher surface with the specular lighting distributed
 * over a larger surface area.
 *
 * See it graphed at:
 * https://www.desmos.com/calculator/pjzk3yafzs
 */
float CalculateNDF(
  in vec3  surfNorm,
  in vec3  halfVector,
  in float roughness
){
  float a = (roughness * roughness);
  float halfAngle = dot(surfNorm, halfVector);
  return (a / (PI * pow((pow(halfAngle, 2.0) * (a - 1.0) + 1.0), 2.0)));
}
/**
 * GGX/Schlick-Beckmann microfacet geometric attenuation.
 *
 * The attenuation is modified by the roughness (input as k)
 * and approximates the influence/amount of microfacets in the surface.
 * A microfacet is a sub-pixel structure that affects light
 * reflection/occlusion.
 */
float CalculateAttenuation(
    in vec3  surfNorm,
    in vec3  vector,
    in float k
){
  float d = max(dot(surfNorm, vector), 0.0);
  return (d / ((d * (1.0 - k)) + k));
}
/**
 * GGX/Schlick-Beckmann attenuation for analytical light sources.
 */
float CalculateAttenuationAnalytical(
    in vec3  surfNorm,
    in vec3  toLight,
    in vec3  toView,
    in float roughness
){
  float k = pow((roughness + 1.0), 2.0) * 0.125;
  float lightAtten = CalculateAttenuation(surfNorm, toLight, k);
  float viewAtten  = CalculateAttenuation(surfNorm, toView, k);
  return (lightAtten * viewAtten);
}
/**
 * GGX/Schlick-Beckmann attenuation for IBL light sources.
 * Uses Disney modification of k to reduce hotness.
 */
float CalculateAttenuationIBL(
    in float roughness,
    in float normDotLight,          // Clamped to [0.0, 1.0]
    in float normDotView            // Clamped to [0.0, 1.0]
){
    float k = pow(roughness, 2.0) * 0.5;
    float lightAtten = (normDotLight / ((normDotLight * (1.0 - k)) + k));
    float viewAtten  = (normDotView / ((normDotView * (1.0 - k)) + k));
    return (lightAtten * viewAtten);
}
/**
 * Calculates the Fresnel reflectivity.
 * The metalic parameter controls the fresnel incident value (fresnel0).
 */
vec3 CalculateFresnel(
    in vec3 surfNorm,
    in vec3 toView,
    in vec3 fresnel0
){
  float d = max(dot(surfNorm, toView), 0.0);
  float p = ((-5.55473 * d) - 6.98316) * d;
  return fresnel0 + ((1.0 - fresnel0) * pow(1.0 - d, 5.0));
}
/**
 * Standard Lambertian diffuse lighting.
 */
vec3 CalculateDiffuse(
    in vec3 albedo
){
    return (albedo * ONE_OVER_PI);
}
/**
 * Cook-Torrance BRDF for analytical light sources.
 */
vec3 CalculateSpecularAnalytical(
    in    vec3  surfNorm,            // Surface normal
    in    vec3  toLight,             // Normalized vector pointing to light source
    in    vec3  toView,              // Normalized vector point to the view/camera
    in    vec3  fresnel0,            // Fresnel incidence value
    inout vec3  sfresnel,            // Final fresnel value used a kS
    in    float roughness            // Roughness parameter (microfacet contribution)
){
    vec3 halfVector = computeHalfVector(toLight, toView);
    float ndf      = CalculateNDF(surfNorm, halfVector, roughness);
    float geoAtten = CalculateAttenuationAnalytical(surfNorm, toLight, toView, roughness);
    sfresnel = CalculateFresnel(surfNorm, toView, fresnel0);
    vec3  numerator   = (sfresnel * ndf * geoAtten);
    float denominator = 4.0 * dot(surfNorm, toLight) * dot(surfNorm, toView);
    return (numerator / denominator);
}
/**
 * Calculates the total light contribution for the analytical light source.
 */
vec3 CalculateLightingAnalytical(
    in vec3  surfNorm,
    in vec3  toLight,
    in vec3  toView,
    in vec3  albedo,
    in float roughness
){
    vec3 fresnel0 = mix(vec3(0.04), albedo, uMetallic);
    vec3 ks       = vec3(0.0);
    vec3 diffuse  = CalculateDiffuse(albedo);
    vec3 specular = CalculateSpecularAnalytical(surfNorm, toLight, toView, fresnel0, ks, roughness);
    vec3 kd       = (1.0 - ks);
    float angle = clamp(dot(surfNorm, toLight), 0.0, 1.0);
    return ((kd * diffuse) + specular) * angle;
}
vec4 compute(const in vec2 sampleUV, const in vec2 pixelUV){
  vec4 pixelAlbedoRGBA = texture( rgbaTex, pixelUV );
  vec4 sampleAlbedoRGBA = texture( rgbaTex, sampleUV );
  float alpha = sampleAlbedoRGBA.a;
  vec3 albedo = sampleAlbedoRGBA.rgb;
  float depth = readDepth(sampleUV);
  vec3 normal = computeNormal(sampleUV, depth);
  vec3 lightDir = normalize(vec3(1,1,1));
  vec3 viewDir = normalize(vec3(0,0,3)-vec3(sampleUV*2.0-1.0, -depth) );
  //vec3 viewDir = normalize(vec3(0,0,1));
  vec3 ambientColor = albedo;
  vec3 diffuseColor = CalculateLightingAnalytical(
    normal,
    lightDir,
    viewDir,
    albedo,
    uRoughness
  );
  vec3 color = ambientColor*uAmbient + diffuseColor*uDiffuse;
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

        # set uniforms
        self.program['uResolution'].value = res
        self.program['uAmbient'].value = float(self.inputs.ambient.get())
        self.program['uDiffuse'].value = float(self.inputs.diffuse.get())
        self.program['uRoughness'].value = float(self.inputs.roughness.get())
        self.program['uMetallic'].value = float(self.inputs.metallic.get())

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
