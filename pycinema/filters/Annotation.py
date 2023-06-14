from pycinema import Filter

import numpy
import sys
import PIL.Image, PIL.ImageFont, PIL.ImageDraw

class Annotation(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'images': [],
            'xy': (20,20),
            'size': 20,
            'spacing': 0,
            'color': 'AUTO',
            'ignore': ['file','id']
          },
          outputs={
            'images': []
          }
        )

    #
    # solution from:
    # https://www.programcreek.com/python/?CodeExample=get+font
    #
    def __get_font(self, size):
        """Attempts to retrieve a reasonably-looking TTF font from the system.

        We don't make much of an effort, but it's what we can reasonably do without
        incorporating additional dependencies for this task.
        """
        font_names = ['../fonts/NotoSansMono-VariableFont_wdth,wght.ttf']
        # if sys.platform == 'win32':
        #     font_names.extend( ['Arial'] )
        # elif sys.platform in ['linux', 'linux2']:
        #     font_names.extend( ['DejaVuSans-Bold', 'DroidSans-Bold'] )
        # elif sys.platform == 'darwin':
        #     font_names.extend( ['Menlo', 'Helvetica'] )

        font = None
        for font_name in font_names:
            try:
                font = PIL.ImageFont.truetype(font_name, size)
                break
            except IOError:
                continue

        return font

    def _update(self):

        images = self.inputs.images.get()

        results = []
        if len(images)<1:
            self.outputs.images.set(results)
            return 1

        textColor = self.inputs.color.get()
        if textColor=='AUTO':
            for image in images:
                if not 'rgba' in image.channels:
                    continue

                mean = images[0].channels['rgba'].mean(axis=(0,1))
                if (mean[0]+mean[1]+mean[2])/3<128 and mean[3]>128:
                    textColor = (255,255,255)
                else:
                    textColor = (0,0,0)
                break

        font = self.__get_font(self.inputs.size.get())

        ignoreList = self.inputs.ignore.get()

        for image in images:
            if not 'rgba' in image.channels:
                results.append( image )
                continue

            rgba = image.channels["rgba"]
            rgbImage = PIL.Image.fromarray( rgba )
            text = ''
            for t in image.meta:
                if t in ignoreList:
                    continue
                m = image.meta[t]
                if isinstance(m, numpy.ndarray):
                    if m.ndim == 0:
                        text = text + ' ' + t+': '+str(m)+'\n'
                    elif m.ndim==1 and len(m)==1:
                        text = text + ' ' + t+': '+str(m[0])+'\n'
                    else:
                        text = text + ' ' + t+': ['+', '.join([str(x) for x in m])+']\n'
                else:
                    text = text + ' ' + t+': '+str(m) + '\n'

            I1 = PIL.ImageDraw.Draw(rgbImage)
            I1.multiline_text(
                self.inputs.xy.get(),
                text,
                fill=textColor,
                font=font,
                spacing=self.inputs.spacing.get()
            )

            outImage = image.copy()
            outImage.channels['rgba'] = numpy.array(rgbImage)
            results.append( outImage )

        self.outputs.images.set(results)

        return 1
