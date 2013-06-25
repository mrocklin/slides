from glob import glob
import os

dots = glob('images/*.dot')
for dot in dots:
    os.system('dot -Tpdf %s -o %s'%(dot, dot[:-3]+'pdf'))
