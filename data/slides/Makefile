filename=main.tex

all: pdf

pdf: $(plots)
	rm main.aux || exit 0
	rm main.bbl || exit 0
	pdflatex -shell-escape ${filename}
	~/Downloads/biber-2.16/bin/biber main.bcf||true
	pdflatex -shell-escape ${filename}
	pdflatex -shell-escape ${filename}
