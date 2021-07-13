all: build/Abgabe.pdf

#continually update .pdf file whenever changes are made to the source files
TexOptions = -pvc \
			 -lualatex \
			 -interaction=nonstopmode \
			 -halt-on-error \
			 -output-directory=build \


.PRECIOUS: build/Abgabe.pdf

#python
plots/Verzoergerung.pdf plots/Marker_Faktor.pdf plots/Lebenszeit.pdf: programm.py | build
	TEXINPUTS=$$(pwd): python programm.py

build/Abgabe.pdf: plots/Verzoergerung.pdf \
				  plots/Marker_Faktor.pdf \
				  plots/Lebenszeit.pdf

build/Abgabe.pdf: FORCE | build
	latexmk $(TexOptions) Abgabe.tex

#a rule without prerequisites or recipe and the target is an nonexistent file 
#	-> make thinks that this target is updated whenever this rule is run
#all other rules depending on FORCE will always have their recipe run
FORCE:

build:
	mkdir -p build/
	mkdir -p plots/	

clean:
	rm -rf build
	rm -rf plots

#to avoid conflict between a target and a file of the same name
#because if there is such a file, the rule would never run, whenever the file is up to date
.PHONY: all clean