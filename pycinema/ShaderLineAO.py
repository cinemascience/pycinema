from .Shader import *
#import cv2
#import numpy

# TODO: actually implement the shader, change the parameters to the ones we need
class ShaderLineAO(Shader):
    def __init__(self):
        super().__init__(['rgbaTex','depthTex'])
        self.addInputPort("images", [])
        self.addInputPort("radius", 2.0)
        self.addInputPort("samples", 32)
        self.addInputPort("scalers", 3)

        self.addOutputPort("images", [])

    def getFragmentShaderCode(self):
        return """
#version 330

uniform sampler2D rgbaTex;
uniform sampler2D depthTex;
uniform float radius;

uniform float diff_area;
uniform int samples;
uniform vec2 resolution;
uniform int scalers;

//uniform int currentLevel;

//uniform sampler2D noiseTex;

//for zoom function
uniform float lineWidth = 2.0;

//gaussian pyramid
uniform int pyramidLevels;

float radiusSS = radius;


//influence of SSAO
uniform float totalStrength = 1.0;


// stregth of occluder in relation to geometry density
uniform float densityWeight = 1.0;


in vec2 uv;
out vec4 color;



//depth

float getDepth(vec2 where, float lod)
{
    return texture2D(depthTex, where).r;
}

float getDepth(vec2 where)
{
    return getDepth(where, 0.0);
}

float getDepth(float lod)
{
    return getDepth(uv, lod);
}

float getDepth()
{
    return getDepth(0.0);
}

//for randNormal
vec2 getTextureSize(sampler2D texture)
{
    vec2 textureSize = textureSize(texture, 0);
    return textureSize;
}


//Normal
vec3 getNormal(vec2 uv) {
    // Lies die Farbe aus der Normal Map
    vec3 colortex = texture(rgbaTex, uv).rgb;

    // Transformiere die Farbe von [0,1] auf [-1,1]
    vec3 normal = colortex * 2.0 - 1.0;

    return normalize(normal);
}

vec3 getNormal(vec2 uv, float lod) {
    // Lies die Farbe aus der Normal Map
    vec3 colortex = texture(rgbaTex, uv).rgb;

    // Transformiere die Farbe von [0,1] auf [-1,1]
    vec3 normal = colortex * 2.0 - 1.0 - lod;

    return normalize(normal);
}


float getZoom()
{
    return texture( rgbaTex, uv ).r;
}



///////////////////////////////////////////
//  LineAO
//////////////////////////////////////////
float computeLineAO(vec2 coord){


    vec4 rgba = texture(rgbaTex,coord);

    vec3 lightPos = vec3(0,10,10);
  

    //openwalnut

    float invSamples = 1.0 / float( samples );

    // Fall-off for SSAO per occluder
    const float falloff = 0.00001;

    
    
    //random normal for reflecting sample rays
    vec2 noiseSize = getTextureSize(rgbaTex);
    vec2 randCoordsNorm = coord * noiseSize.x;
    vec3 randNormal = vec3(fract(sin(dot(randCoordsNorm, vec2(12.9898, 78.233))) * 43758.5453), 
                        fract(sin(dot(randCoordsNorm, vec2(23.123, 45.678))) * 98365.1234), 
                        fract(sin(dot(randCoordsNorm, vec2(34.567, 89.012))) * 29384.9876));
    randNormal = normalize(randNormal * 2.0 - vec3(1.0));
    
   //vec3 randNormal = normalize( ( texture( noiseTex, coord * int(noiseSize.x) ).xyz * 2.0 ) - vec3( 1.0 ) ); 


    //current pixels normal and depth
    float currentPixelDepth = getDepth( coord );
    vec3 currentPixelSample = getNormal( coord ).xyz;
    //vec3 currentPixelSample = computeNormal( coord, readDepth(coord) ).xyz;
     
    //current fragment coords
    vec3 ep = vec3( coord.xy, currentPixelDepth);

    //normal of current fragment
    vec3 normal = currentPixelSample.xyz;

    //invariant for zooming 
   
    float maxPixels = max( float( resolution.x ), float( resolution.y ) );
    //radiusSS = ( getZoom() * radius / maxPixels ) / (1.0 - currentPixelDepth );
    radiusSS = ( radius / maxPixels ) / (1.0 - currentPixelDepth );

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
        #define radScaleMin 1.5
        radiusScaler = radScaleMin + l;

        //get samples and check for Occluders
        int numSamplesAdded = 0;

        for( int i = 0; i < samples; ++i){
        
            //random normal from noise texture
            
            vec2 randCoordsSphere = vec2(float(i) / float(samples), float(l + 1) / float(scalers));
            vec3 randSphereNormal = vec3(fract(sin(dot(randCoordsSphere,  vec2(12.9898, 78.233))) * 43758.5453), 
                                        fract(sin(dot(randCoordsSphere, vec2(23.123, 45.678))) * 98365.1234), 
                                        fract(sin(dot(randCoordsSphere, vec2(34.567, 89.012))) * 29384.9876));
            randSphereNormal = normalize(randSphereNormal * 2.0 - vec3(1.0));
            
            
            //vec3 randSphereNormal = ( texture( noiseTex, vec2( float( i ) / float( samples ), float( l + 1 ) / float( scalers ) ) ).rgb * 2.0 ) - vec3( 1.0 );


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

            float lod = 0.0;
            lod = float(l);

            //gausspyramid for Level of Detail
            
            occluderDepth = getDepth( hemispherePoint.xy, lod);
            occluderNormal = getNormal( hemispherePoint.xy, lod ).xyz;
            depthDifference = currentPixelDepth - occluderDepth;

            //difference between the normals as a weight -> how much occludes fragment
            float pointDiffuse = max( dot( hemisphereVector, normal ), 0.0 ); 

            //spielt Rolle bei brightness:
            //diffuse reflected light
                    //#ifdef OccluderLight
                    //vec3 t= getTangent( hemispherePoint.xy, lod ).xyz;
                    //vec3 newnorm = normalize( cross( normalize( cross( t, normalize( hemisphereVector ) ) ), t ) );
                    //float occluderDiffuse = max( dot( newnorm, lightPos.xyz ), 0.0);

                    //#else
            
            //disable effect
            float occluderDiffuse = 0.0;
            //#endif
        

            //specular reflection
            vec3 H = normalize( lightPos.xyz + normalize( hemisphereVector ) );
            float occluderSpecular = pow( max( dot( H, occluderNormal ), 0.0 ), 100.0);

            normalDifference = pointDiffuse * ( occluderSpecular + occluderDiffuse );
            normalDifference= 1.5 - normalDifference;


            //shadowiness

            float SCALER = 1.0 - ( l / (float (scalers - 1.0 ) ) );
            float densityInfluence = SCALER * SCALER * densityWeight;

            float densityWeight = 1.0 - smoothstep( falloff, densityInfluence, depthDifference );


            // reduce occlusion if occluder is far away
            occlusionStep += normalDifference * densityWeight * step( falloff, depthDifference );
            //occlusionStep += normalDifference * densityWeight;
            //occlusionStep = 0.0;
        }

        occlusion += ( 1.0 / float( numSamplesAdded ) ) * occlusionStep;
    }

    float occlusionScalerFactor = 1.0 / ( scalers );
    occlusionScalerFactor *= totalStrength;

    //output result
    return clamp( ( 1.0 - ( occlusionScalerFactor * occlusion  ) ), 0.0 , 1.0 );  
    //return randSphereNormal.x;
}

void main(){

    vec4 rgba = texture(rgbaTex, uv);
    float c = computeLineAO(uv);

    //color = vec4(c, rgba.a);
    color = vec4(mix( vec3(0), rgba.rgb, c), rgba.a);
    //color = vec4(vec3(rgba.rgb*c), 1.0);
}

"""

    def render(self,image):
        # update texture
        self.rgbaTex.write(image.channels['rgba'].tobytes())
        #self.depthTex.write(image.channels['depth'].tobytes())

        # render
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # read framebuffer
        outImage = image.copy()
        outImage.channels['rgba'] = self.readFramebuffer()

        return outImage

#    def downsample(self,image):
        image_data= numpy.array(image)
        # Verkleinert das Bild auf die Hälfte seiner Größe
        return cv2.pyrDown(image_data)
    
#    def create_gaussian_pyramid(self, image, levels):   
        pyramid = [image]
        for i in range(levels):
            current_level = i  # Setzen Sie dies auf das aktuelle Level Ihrer Gauß-Pyramide
            self.program['currentLevel'].value = current_level
            # Glätte das Bild mit dem Shader
            smoothed_image = self.render(pyramid[-1])
            # Skaliere das Bild herunter
            downsampled_image = self.downsample(smoothed_image)
            pyramid.append(downsampled_image)
        return pyramid
        
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
        
        self.program['radius'].value = float(self.inputs.radius.get())
        self.program['samples'].value = int(self.inputs.samples.get())
        self.program['scalers'].value = int(self.inputs.scalers.get())
        self.program['resolution'].value = res
        
       

        
        # create textures
        self.rgbaTex = self.createTexture(0,res,shape[2],dtype='f1')
        self.depthTex = self.createTexture(1,res,1,dtype='f4')
        #self.noiseTex = self.createNoiseTexture(2, res, 3)
        

        #self.create_gaussian_pyramid(images, 8)
            

        for image in images:
            results.append( self.render(image) )


        self.rgbaTex.release()
        self.depthTex.release()
        

        self.outputs.images.set(results)

        return 1
