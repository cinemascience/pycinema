from .Shader import *

# TODO: actually implement the shader, change the parameters to the ones we need
class ShaderPointAO(Shader):
    def __init__(self):
        super().__init__(['rgbaTex','depthTex'])
        self.addInputPort("images", [])
        self.addInputPort("radius", 2.0)
        self.addInputPort("samples", 32)
        self.addInputPort("scalers", 3)
        self.addInputPort("zoom", 60.0)
        self.addInputPort("diff", 0.5)
        self.addOutputPort("images", [])

    def getFragmentShaderCode(self):
        return """
#version 330

uniform samp
niform sampler2D depthTex;
uniform float radius;

uniform float diff_area;
uniform int samples;
uniform float zoom;
uniform vec2 resolution;
uniform int scalers;
uniform sampler2D noiseSampler;


//for zoom function
uniform float lineWidth = 2.0;

float radiusSS;


//influence of SSAO
uniform float totalStrength = 1.0;


// stregth of occluder in relation to geometry density
uniform float densityWeight = 1.0;


in vec2 uv;
out vec4 color;


float readDepth(vec2 coord){
    //float d = texture(depthTex,coord).r;
    //return isnan(d) ? 1.0 : d;
    return texture(depthTex,coord).r;
}

vec3 computeNormal(in vec2 uv, in float depth){
    vec2 pixelSize = 1./resolution;
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


//Source: https://github.com/ashima/webgl-noise/blob/3e2528debc5e5e51a35bf154e7d27c8f98078f8a/src/noise2D.glsl


vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(vec2 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 permute(vec3 x) {
    return mod289(((x*34.0)+10.0)*x);
}

float snoise(vec2 v)
  {
    const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                        0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                        -0.577350269189626,  // -1.0 + 2.0 * C.x
                        0.024390243902439); // 1.0 / 41.0
    // First corner
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);

    // Other corners
    vec2 i1;
    //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
    //i1.y = 1.0 - i1.x;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    // x0 = x0 - 0.0 + 0.0 * C.xx ;
    // x1 = x0 - i1 + 1.0 * C.xx ;
    // x2 = x0 - 1.0 + 2.0 * C.xx ;
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;

    // Permutations
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
	    + i.x + vec3(0.0, i1.x, 1.0 ));

    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;

    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;

    // Normalise gradients implicitly by scaling m
    // Approximation of: m *= inversesqrt( a0*a0 + h*h );
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

    // Compute final noise value at P
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

//zoom function for lineAO

float getZoom(vec2 coord){

    
    float halfFov = radians(zoom) / 2.0;
    float screenHeight = 2.0 * readDepth(coord) * tan(halfFov);
    float aspect = resolution.y/resolution.x;
    float zoomFactor = lineWidth / aspect;
    
    return zoomFactor;

}


//LineAO

float computeLineAO(vec2 coord){

    vec4 rgba = texture(rgbaTex,coord);
    float depth = readDepth(coord);

    vec2 pixelSize = 1./resolution;
    
  
    vec3 lightPos = vec3(0,0,1);
  

    //openwalnut

    float invSamples = 1.0 / float( samples );

    // Fall-off for SSAO per occluder
    const float falloff = 0.0001;

    //random normal for reflecting sample rays
    vec3 randNormal = normalize( (texture( noiseSampler, coord * snoise(coord) ).xyz * 2.0) - vec3(1.0)  );
  

    //current pixels normal and depth
    vec3 currentPixelSample = computeNormal( coord, depth).xyz;
    float currentPixelDepth = readDepth( coord );

    //current fragment coords
    vec3 ep = vec3( coord.xy, currentPixelDepth);

    //normal of current fragment
    vec3 normal = currentPixelSample.xyz;

    //invariant for zooming
    float maxPixels = max( float( resolution.x ), float( resolution.y ) );
    radiusSS = ( getZoom(coord) * radius / maxPixels ) / (1.0 - currentPixelDepth );

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
    for( int l = 0; 1 < scalers; ++l)
    {
    float occlusionStep = 0.0;

    //diffrent from paper
    #define radScaleMin 1.5
    radiusScaler += radScaleMin + 1;

    //get samples and check for Occluders
    int numSamplesAdded = 0;

    for( int i = 0; 1 < samples; ++i){
        
        //random normal from noise texture
        vec3 randSphereNormal = ( texture( noiseSampler, vec2( float( i ) / float( samples ), float( l + 1) / float( scalers ) ) ).rgb * 2.0 ) - vec3( 1.0 );
        

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
        
        //gausspyramid can be used for Level of Detail
        //...

        //occluderDepth = readDepth( hemispherePoint.xy);
        occluderNormal = computeNormal( hemispherePoint.xy, currentPixelDepth ).xyz;

        //difference between the normals as a weight -> how much occludes fragment
        float pointDiffuse = max( dot( hemisphereVector, normal ), 0.0 ); 

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

        float SCALER = 1.0 - ( 1 / (float (scalers - 1 ) ) );
        float densityInfluence = SCALER * SCALER * densityWeight;

        float densityWeight 0 1.0 - smoothstep( falloff, densityInfluence, depthDifference );

       // float e0 = falloff;
       // float e1 = densityInfluence;
       // float r = ( depthDifference - e0 ) / ( e1 - e0 );
       // float desityWeight = 1.0 - smoothstep ( 0.0, 1.0, r);

        // reduce occlusion if occluder is far away
        occlusionStep += normalDifference * densityWeight * step( falloff, depthDifference );
    

        occlusion += ( 1.0 / float( numSamplesAdded ) ) * occlusionStep;
    }
    
    float occlusionScalerFactor = 1.0 / ( scalers );
    occlusionScalerFactor *= totalStrength;

    //output result
    return clamp( 1.0 - ( occlusionScalerFactor * occlusion ) , 0 , 1 );

}


}

void main(){


    float c = computeLineAO(uv);

   vec4 rgba = texture(rgbaTex, uv);

    //color = vec4(c, rgba.a);
    color = vec4(mix( vec3(0), rgba.rgb, c), rgba.a);
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
        self.program['scalers'].value = int(self.inputs.scalers.get())
        
       

        # create textures
        self.rgbaTex = self.createTexture(0,res,shape[2],dtype='f1')
        self.depthTex = self.createTexture(1,res,1,dtype='f4')

        for image in images:
            results.append( self.render(image) )

        self.rgbaTex.release()
        self.depthTex.release()

        self.outputs.images.set(results)

        return 1
