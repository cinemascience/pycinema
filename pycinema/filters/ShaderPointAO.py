from .Shader import *

class ShaderPointAO(Shader):
    def __init__(self):
        super().__init__(
          inputs={
            'images': [],
            'radius': 1.5,
            'samples': 32,
            'scalers': 3,
            'densityweight': 1.0,
            'totalStrength': 1.0
          },
          outputs={
            'images': []
          },
          textures=['rgbaTex','depthTex','noiseTex']
        )

    def getFragmentShaderCode(self):
        return """
#version 330

uniform sampler2D rgbaTex;
uniform sampler2D depthTex;
uniform sampler2D noiseTex;
uniform float radius;

uniform float diff_area;
uniform int samples;
uniform vec2 resolution;
uniform int scalers;
uniform float densityweight;

//uniform int currentLevel;

//for zoom function
uniform float lineWidth = 2.0;

float radiusSS = radius;


//influence of SSAO
uniform float totalStrength;


in vec2 uv;
out vec4 color;



//depth

float getDepth(vec2 where)
{
    return texture(depthTex, where).r;
}

//for randNormal
vec2 getTextureSize(sampler2D texture)
{
    vec2 textureSize = textureSize(texture, 0);
    return textureSize;
}


//Normal

vec3 computeNormal(in vec2 uv, in float depth){
  vec2 pixelSize = 1./resolution;
  vec3 eps = vec3( pixelSize.x, pixelSize.y, 0 );
  float depthN = getDepth(uv.xy + eps.zy);
  float depthS = getDepth(uv.xy - eps.zy);
  float depthE = getDepth(uv.xy + eps.xz);
  float depthW = getDepth(uv.xy - eps.xz);
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



//  PointAO

float computePointAO(vec2 coord){


    vec4 rgba = texture(rgbaTex,coord);

    vec3 lightPos = vec3(0,10,10);


    //openwalnut

    // Fall-off for SSAO per occluder
    const float falloff = 0.00001;



    //random normal for reflecting sample rays
    vec2 noiseSize = getTextureSize(noiseTex);

    //vec2 randCoordsNorm = coord * noiseSize.x;
    //vec3 randNormal = vec3(fract(sin(dot(randCoordsNorm, vec2(12.9898, 78.233))) * 43758.5453),
    //                    fract(sin(dot(randCoordsNorm, vec2(23.123, 45.678))) * 98365.1234),
    //                    fract(sin(dot(randCoordsNorm, vec2(34.567, 89.012))) * 29384.9876));
    //randNormal = normalize(randNormal * 2.0 - vec3(1.0));

    vec3 randNormal = normalize( ( texture( noiseTex, coord * int(noiseSize.x) ).xyz * 2.0 ) - vec3( 1.0 ) );


    //current pixels normal and depth
    float currentPixelDepth = getDepth( coord );
    vec3 currentPixelSample = computeNormal( coord, getDepth(coord) ).xyz;

    //current fragment coords
    vec3 ep = vec3( coord.xy, currentPixelDepth);

    //normal of current fragment
    vec3 normal = currentPixelSample.xyz;

    float maxPixels = max( float( resolution.x ), float( resolution.y ) );



    //some temoraries needed inside the loop
    vec3 ray;
    vec3 hemispherePoint;
    vec3 occluderNormal;

    float occluderDepth;
    float depthDifference;
    float normalDifference;

    float occlusion = 0.0;
    float radiusScaler = 0.0;

    //sample for different radii
    for( int l = 0; l < scalers; ++l)
    {
        float occlusionStep = 0.0;

        //diffrent from paper
        radiusScaler += radius + l;

        //get samples and check for Occluders
        int numSamplesAdded = 0;

        for( int i = 0; i < samples; ++i){

            //random normal from noise texture

            //vec2 randCoordsSphere = vec2(float(i) / float(samples), float(l + 1) / float(scalers));
            //vec3 randSphereNormal = vec3(fract(sin(dot(randCoordsSphere,  vec2(12.9898, 78.233))) * 43758.5453),
            //                            fract(sin(dot(randCoordsSphere, vec2(23.123, 45.678))) * 98365.1234),
            //                            fract(sin(dot(randCoordsSphere, vec2(34.567, 89.012))) * 29384.9876));
            //randSphereNormal = normalize(randSphereNormal * 2.0 - vec3(1.0));



            vec3 randSphereNormal = ( texture( noiseTex, vec2( float( i ) / float( samples ), float( l + 1 ) / float( scalers ) ) ).rgb * 2.0 ) - vec3( 1.0 );

            //radius corresponds to (1 / (1 - dj(P))) * r_0 * (j^2 + j * r_0)
            float depthScaling = 1.0 / (1.0 - currentPixelDepth);
            float distanceLevelScaling = (l * l + l * radius);
            radiusSS = depthScaling * radius * distanceLevelScaling / maxPixels;

            vec3 hemisphereVector = reflect( randSphereNormal, randNormal );
            ray = radiusScaler * radiusSS * hemisphereVector;
            ray = sign( dot( ray, normal ) ) * ray;

            //point in texture space on the hemisphere
            hemispherePoint = ray + ep;

            if( ( hemispherePoint.x < 0.0 ) || ( hemispherePoint.x > 1.0 ) ||
                ( hemispherePoint.y < 0.0 ) || ( hemispherePoint.y > 1.0 ) )
            {
                continue;
            }

            //count Samples used
            numSamplesAdded++;


            occluderDepth = getDepth( hemispherePoint.xy );
            occluderNormal = computeNormal( hemispherePoint.xy, occluderDepth).xyz;
            depthDifference = currentPixelDepth - occluderDepth;

            //difference between the normals as a weight -> how much occludes fragment
            float pointDiffuse = max( dot( hemisphereVector, normal ), 0.0 );



            //depth-based weighting function
            vec3 H = normalize( hemisphereVector );
            float occluderweight = pow( max( dot( H, occluderNormal ), 0.0 ), 100.0);

            normalDifference = pointDiffuse * occluderweight;
            normalDifference = 1.5 - normalDifference;


            //shadowiness

            float SCALER = 1.0 - ( l / (float (scalers - 1.0 ) ) );
            float densityInfluence = SCALER * SCALER * densityweight;

            float lineDensityWeight = 1.0 - smoothstep( falloff, densityInfluence, depthDifference );


            // reduce occlusion if occluder is far away
            occlusionStep += normalDifference * lineDensityWeight * step( falloff, depthDifference );

        }

        occlusion += ( 1.0 / float( numSamplesAdded ) ) * occlusionStep;
    }

    float occlusionScalerFactor = 1.0 / ( scalers );
    occlusionScalerFactor *= totalStrength;

    //output result
    return clamp( ( 1.0 - ( occlusionScalerFactor * occlusion  ) ), 0.0 , 1.0 );
}

void main(){

    vec4 rgba = texture(rgbaTex, uv);
    float c = computePointAO(uv);

    color = vec4(mix( vec3(0), rgba.rgb, c), rgba.a);

}

"""

    def create_noise_texture(self, resolution):
        noise_data = numpy.random.rand(resolution[0], resolution[1], 3)
        noise_data = (noise_data * 255).astype(numpy.uint8)
        self.noise_data = noise_data
        self.noise_data_resolution = resolution

    def render(self,image):
        # update framebuffer and textures
        self.initFramebuffer(image.resolution)
        self.updateTexture(0,image.getChannel('rgba'))
        self.updateTexture(1,image.getChannel('depth'))
        if self.noise_data_resolution != image.resolution:
            self.create_noise_texture(image.resolution)
        self.updateTexture(2,self.noise_data,True)

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
        self.program['scalers'].value = int(self.inputs.scalers.get())
        self.program['densityweight'].value = int(self.inputs.densityweight.get())
        self.program['totalStrength'].value = int(self.inputs.totalStrength.get())

        # render images
        try:
          for image in images:
              results.append( self.render(image) )
        except:
          self.outputs.images.set(images)
          return 1

        self.outputs.images.set(results)

        return 1
