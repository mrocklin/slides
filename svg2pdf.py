from glob import glob
import os

svgs = glob('images/*.svg')
for svg in svgs:
    os.system('rsvg-convert -f pdf %s > %s'%(svg, svg[:-3]+'pdf'))
