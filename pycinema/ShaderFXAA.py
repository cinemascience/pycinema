from .Shader import *

class ShaderFXAA(Shader):
    def __init__(self):
        super().__init__(['rgbaTex'], ['uv','v_rgbNW','v_rgbNE','v_rgbSW','v_rgbSE','v_rgbM'])
        self.addInputPort("images", [])
        self.addOutputPort("images", [])

    def getVertexShaderCode(self):
        return """
#version 330

uniform vec2 resolution;

in vec2 position;

out vec2 uv;
out vec2 v_rgbNW;
out vec2 v_rgbNE;
out vec2 v_rgbSW;
out vec2 v_rgbSE;
out vec2 v_rgbM;

void main() {
    uv = position/2.0+0.5;

    vec2 fragCoord = uv * resolution;
    vec2 inverseVP = 1.0 / resolution.xy;
    v_rgbNW = (fragCoord + vec2(-1.0, -1.0)) * inverseVP;
    v_rgbNE = (fragCoord + vec2(1.0, -1.0)) * inverseVP;
    v_rgbSW = (fragCoord + vec2(-1.0, 1.0)) * inverseVP;
    v_rgbSE = (fragCoord + vec2(1.0, 1.0)) * inverseVP;
    v_rgbM = vec2(fragCoord * inverseVP);

    gl_Position = vec4(position,0,1);
}
        """

    def getFragmentShaderCode(self):
        return """
#version 330

in vec2 uv;

in vec2 v_rgbNW;
in vec2 v_rgbNE;
in vec2 v_rgbSW;
in vec2 v_rgbSE;
in vec2 v_rgbM;

out vec4 outcolor;

uniform sampler2D rgbaTex;
uniform vec2 resolution;

#ifndef FXAA_REDUCE_MIN
    #define FXAA_REDUCE_MIN   (1.0/ 128.0)
#endif

#ifndef FXAA_REDUCE_MUL
    #define FXAA_REDUCE_MUL   (1.0 / 8.0)
#endif

#ifndef FXAA_SPAN_MAX
    #define FXAA_SPAN_MAX     8.0
#endif

vec4 fxaa(sampler2D tex, vec2 fragCoord, vec2 resolution,
            vec2 v_rgbNW, vec2 v_rgbNE,
            vec2 v_rgbSW, vec2 v_rgbSE,
            vec2 v_rgbM) {
    vec4 color;
    mediump vec2 inverseVP = vec2(1.0 / resolution.x, 1.0 / resolution.y);
    vec3 rgbNW = texture2D(tex, v_rgbNW).xyz;
    vec3 rgbNE = texture2D(tex, v_rgbNE).xyz;
    vec3 rgbSW = texture2D(tex, v_rgbSW).xyz;
    vec3 rgbSE = texture2D(tex, v_rgbSE).xyz;
    vec4 texColor = texture2D(tex, v_rgbM);
    vec3 rgbM  = texColor.xyz;
    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM,  luma);
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    mediump vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) *
                          (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);

    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
              max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
              dir * rcpDirMin)) * inverseVP;

    vec3 rgbA = 0.5 * (
        texture2D(tex, fragCoord * inverseVP + dir * (1.0 / 3.0 - 0.5)).xyz +
        texture2D(tex, fragCoord * inverseVP + dir * (2.0 / 3.0 - 0.5)).xyz);

    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture2D(tex, fragCoord * inverseVP + dir * -0.5).xyz +
        texture2D(tex, fragCoord * inverseVP + dir * 0.5).xyz);

    float lumaB = dot(rgbB, luma);
    if ((lumaB < lumaMin) || (lumaB > lumaMax))
        color = vec4(rgbA, texColor.a);
    else
        color = vec4(rgbB, texColor.a);

    return color;
}

void main() {
    vec2 fragCoord = uv * resolution;
    outcolor = fxaa(rgbaTex, fragCoord, resolution, v_rgbNW, v_rgbNE, v_rgbSW, v_rgbSE, v_rgbM);
}

"""

    def render(self,image):
        rgba = image.channels['rgba']

        # create texture
        self.rgbaTex.write(rgba.tobytes())

        # render
        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # read pixels
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
        if not 'rgba' in image0.channels:
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

        # create textures
        self.rgbaTex = self.createTexture(0,res,shape[2],dtype='f1')

        for image in images:
            results.append( self.render(image) )

        self.rgbaTex.release()

        self.outputs.images.set(results)

        return 1
