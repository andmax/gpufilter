PDF := algs456-perf.pdf bicubic-interpolation.pdf gaussian-blur.pdf filter-decay.pdf 4k-stages.pdf

all: $(PDF)

SVG := algs456-perf.svg bicubic-interpolation.svg gaussian-blur.svg filter-decay.svg 4k-stages.svg

%.pdf: %.svg
	cat $< | sed 's/linecap="butt"/linecap="round"/g' | sed 's/linejoin="miter"/linejoin="round"/g' | sed 's/"-0.50em"/"-3.5pt" font-size="9.0"/g' | sed 's/font-size="9.0"/font-size="8.0"/g' > $<.new

	mv -f $<.new $<
	inkscape -z -f $< -A $@

$(SVG): plots.gpi
	gnuplot $<

clean:
	rm -f $(SVG)
	rm -f $(PDF)
