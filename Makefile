slides.tex: slides.md my.beamer kalman.f90
	python scripts/include.py slides.md slides2.md
	python scripts/dot2pdf.py
	pandoc -t beamer slides2.md -o slides.tex --standalone --template=my.beamer --variable fontsize=8pt -H tex/preamble-extra.tex
	python dollar.py slides.tex slides.tex
	python scripts/svg2pdf.py

images/pdfs: images/*.svg
	python svg2pdf.py

pdf: slides.tex images/pdfs
	pdflatex slides.tex

publish: pdf
	scp slides.pdf mrocklin@belvedere.cs.uchicago.edu:html/storage/defense-slides.pdf
