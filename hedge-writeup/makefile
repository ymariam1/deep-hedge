all: deep-hedge.pdf

deep-hedge.pdf: deep-hedge.tex
	latexmk -pdf -silent $<

clean:
	latexmk -C
