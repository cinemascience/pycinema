from .Shader import *

import pycinema

import numpy
import moderngl
import PIL

class ShaderDemoScene(Shader):
    def __init__(self):
        super().__init__(
          inputs={
            'resolution': (256,256),
            'phi_samples': (0,360,360),
            'theta_samples': (20,20,1),
            'time_samples': (0,0,1),
            'objects': (1,1,1),
          },
          outputs={
            'images': []
          }
        )

    def getVertexShaderCode(self):
        return """
#version 330

in vec2 position;
out vec2 uv;

void main(){
    uv = vec2(1,-1)*position;
    gl_Position = vec4(position,0,1);
}
"""

    def getFragmentShaderCode(self):
        return """
#version 330

in vec2 uv;
layout(location=0) out vec4 outColor;
layout(location=1) out float outDepth;
layout(location=2) out float outId;
layout(location=3) out float outY;
uniform vec2 iResolution;
uniform float iTime;
uniform vec3 iObjects;
uniform float iPhi;
uniform float iTheta;
uniform float NAN;

const int MAX_MARCHING_STEPS = 255;
const float MIN_DIST = 0.1;
const float MAX_DIST = 50.0;
const float DELTA_DIST = MAX_DIST - MIN_DIST;
const float EPSILON = 0.001;

float planeSDF(vec3 p) {
    return abs(p.y);
}

float sphereSDF(vec3 p, float r) {
    return length(p) - r;
}

vec2 compare(vec2 hit, float d, float id){
    return hit.x<d ? hit : vec2(d,id);
}

vec2 sceneSDF(vec3 p) {
    vec2 hit = vec2(MAX_DIST,-1);

    if(iObjects.x>0.5)
      hit = compare(hit, planeSDF(p), 0);

    if(iObjects.y>0.5)
      hit = compare(hit, sphereSDF(p-vec3(0.7,0.25,0.7), 0.2), 1);
      //hit = compare(hit, sphereSDF(p-2.0*vec3(cos(iTime),0.25,sin(iTime)), 0.25), 1);

    if(iObjects.z>0.5)
      hit = compare(hit, sphereSDF(p-vec3(0,1,0),max(0.1,1.0-iTime)), 2);

    return hit;
}

vec3 march(vec3 ro, vec3 rd, float tmin, float tmax) {
    vec3 hit = vec3(0,-1,tmin);
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        hit.xy = sceneSDF(ro + hit.z * rd);
        if (hit.x < EPSILON) {
            return hit;
        }
        hit.z += hit.x;
        if (hit.z >= tmax) {
            return hit;
        }
    }
    return hit;
}

vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)).x - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)).x,
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)).x - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)).x,
        sceneSDF(vec3(p.x, p.y, p.z    + EPSILON)).x - sceneSDF(vec3(p.x, p.y, p.z - EPSILON)).x
    ));
}

vec3 phongBRDF(vec3 lightDir, vec3 rayDir, vec3 normal, vec3 diff, vec3 spec, float shininess) {
    vec3 color = diff;
    vec3 reflectDir = reflect(-lightDir, normal);
    float specDot = max(dot(reflectDir, rayDir), 0.0);
    color += pow(specDot, shininess) * spec;
    return color;
}

float softshadow(vec3 ro, vec3 rd, float tmin, float tmax)
{
    float res = 1.0;
    vec3 hit = vec3(0,-1,tmin);
    for (int i = 0; i < 16; i++) {
        hit.xy = sceneSDF(ro + hit.z * rd);
        res = min( res, 8.0*hit.x/hit.z );
        hit.z += hit.x;
    }
    return clamp( res, 0.0, 1.0 );
}

void main() {
    vec2 fragCoord = (uv*0.5+0.5)*iResolution;
    vec3 focal = normalize(vec3(
        cos(iPhi)*sin(iTheta),
        cos(iTheta),
        sin(iPhi)*sin(iTheta)
    ));

    float aspect = iResolution.x/iResolution.y;
    vec3 rayDir = -normalize(focal);
    vec3 up = vec3(0,1,0);
    vec3 right = normalize(cross(rayDir,up));
    vec3 up2 = -normalize(cross(rayDir,right));

    float scale = 2.0;
    vec3 origin = 15.0*focal + aspect*right*uv.x*scale + up2*uv.y*scale;

    vec3 hit = march(origin, rayDir, MIN_DIST, MAX_DIST);

    if (hit.z > MAX_DIST - EPSILON) {
        outColor = vec4(0);
        outDepth = 1.0;
        outId = NAN;
        outY = NAN;
        return;
    }

    // The closest point on the surface to the eyepoint along the view ray
    vec3 p = origin + hit.z * rayDir;
    vec3 lightDir = normalize(vec3(1,1,0));
    vec3 normal = estimateNormal(p);

    vec3 materialColor = hit.y > 1.5 ? vec3(0.8,0,0) : hit.y > 0.5 ? vec3(0,0.8,0) :  vec3(0.3 + 0.1*mod( floor(1.0*p.z) + floor(1.0*p.x), 2.0));

    vec3 radiance = vec3(0);
    float irradiance = max(dot(lightDir, normal), 0.0);
    vec3 brdf = phongBRDF(lightDir, rayDir, normal, materialColor, vec3(1), 1000.0);
    radiance += brdf * irradiance * vec3(1);
    radiance *= softshadow(p, lightDir, MIN_DIST, MAX_DIST);

    outColor = vec4(
        pow(radiance, vec3(1.0 / 2.2) ), // gamma correction
        1.0
    );
    outDepth = (hit.z-MIN_DIST)/DELTA_DIST;
    outId = hit.y;
    outY = p.y;
}
        """

    def render(self,phi,theta,time,objects_meta):

        Shader.fbo.clear(0.0, 0.0, 0.0, 1.0)

        phi_rad = phi/360.0*2.0*numpy.pi
        theta_rad = (90-theta)/180.0*numpy.pi

        # render
        self.program['iObjects'].value = self.inputs.objects.get()
        self.program['iTime'].value = time
        self.program['iPhi'].value = phi_rad
        self.program['iTheta'].value = theta_rad
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # create output image
        return pycinema.Image(
            {
                'rgba': self.readFramebuffer(0,4,numpy.uint8),
                'depth': self.readFramebuffer(1,1,numpy.float32),
                'id': self.readFramebuffer(2,1,numpy.float32),
                'y': self.readFramebuffer(3,1,numpy.float32)
            },
            {
                'time': time,
                'phi': phi,
                'theta': theta,
                'object_id': objects_meta
            }
        )

    def getRange(self,bounds,mod=0):
        if numpy.isscalar(bounds):
            return [bounds]

        array = numpy.arange(bounds[0],bounds[1]+bounds[2],bounds[2])
        if mod != 0:
            array = list(map(lambda x: x % mod, array))
            array = numpy.unique(array)

        return array

    def _update(self):

        phi_samples = self.getRange(self.inputs.phi_samples.get(),360)
        theta_samples = self.getRange(self.inputs.theta_samples.get(),360)
        time_samples = self.getRange(self.inputs.time_samples.get())

        # create framebuffer
        res = self.inputs.resolution.get()
        self.initFramebuffer(res,[4,1,1,1],['f1','f4','f4','f4'])

        self.program['iResolution'].value = res
        self.program['NAN'].value = numpy.nan

        objects = self.inputs.objects.get()
        objects_meta = ['p','s0','s1']
        objects_meta = [objects_meta[i] for i in range(3) if objects[i]]
        objects_meta = '+'.join(objects_meta)

        results = []

        for time in time_samples:
            for theta in theta_samples:
                for phi in phi_samples:
                    results.append(
                        self.render(phi, theta, time, objects_meta)
                    )

        self.outputs.images.set(results);

        return 1;
