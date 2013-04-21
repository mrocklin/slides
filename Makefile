slides.tex: slides.md my.beamer
	pandoc -t beamer slides.md -o slides.tex --standalone --template=my.beamer --variable fontsize=8pt
	python dollar.py slides.tex slides.tex

images/pdfs: images/*.svg
	python svg2pdf.py

pdf: slides.tex images/pdfs
	pdflatex slides.tex

publish: pdf
	scp slides.pdf mrocklin@belvedere.cs.uchicago.edu:html/storage/sandia-slides.pdf
