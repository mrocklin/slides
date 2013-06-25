import re
import os

regex = "include .*\(([\w-]*\.\w*)\)"

def include(s):
    match = re.search(regex, s)
    if not match:
        return s
    else:
        fn = match.groups()[0]
        if not os.path.exists(fn):
            s = s.replace(match.group(),
                    'Not found: ' + match.group()[len('include'):])
        else:
            with open(fn) as f:
                newtext = f.read()
                s = s.replace(match.group(), newtext)
        return include(s)

def process_file(instream, outstream):
    outstream.write(include(instream.read()))

if __name__ == '__main__':
    from sys import argv, stdin, stdout
    if len(argv) == 1:
        process_file(stdin, stdout)
    elif len(argv) == 3:
        with open(argv[1]) as instream:
            with open(argv[2], 'w') as outstream:
                process_file(instream, outstream)
    else:
        print "usage: python include.py inputfile outputfile"
