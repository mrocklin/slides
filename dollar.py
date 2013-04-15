from sys import argv

if not len(argv) == 3:
    print "Usage: python %s infile outfile" % argv[0]
    exit(0)

with open(argv[1]) as infile:
    s = infile.read()

s = s.replace(r"\$", "$")

with open(argv[2], 'w') as outfile:
    outfile.write(s)
