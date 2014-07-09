slides: pydata-sv-2014.ipynb
	ipython nbconvert --to slides --post serve pydata-sv-2014.ipynb

slides.tex: slides.md my.beamer
	pandoc -t beamer slides.md -o slides.tex --standalone --template=my.beamer --variable fontsize=8pt
	python dollar.py slides.tex slides.tex

pdf: slides.tex
	pdflatex slides.tex

publish: pdf
	scp slides.pdf LOCATION
