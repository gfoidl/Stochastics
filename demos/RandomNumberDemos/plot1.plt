# http://psy.swansea.ac.uk/staff/carter/gnuplot/gnuplot_frequency.htm

reset

set datafile separator ";"

set term pngcairo size 1128,800 enhanced
set output 'distribution1.png'

set multiplot layout 3,2

set style line 1 linecolor  rgb "blue"
set style line 2 linecolor  rgb "green"
set style line 3 linecolor  rgb "red"

# Add a vertical dotted line at x=0 to show centre (mean) of distribution.
set yzeroaxis

bin_width = 0.05;
bin_number(x) = floor(x / bin_width)
rounded(x) = bin_width * bin_number(x)

set title 'Numbers'
set autoscale x
plot 'results.txt' using 1 with lines ls 1 title 'System.Random', \
     'results.txt' using 2 with lines ls 2 title 'Uniform'

set title 'Distribution'
plot 'results.txt' using (rounded($1)) smooth frequency with lines ls 1 title 'System.Random', \
     'results.txt' using (rounded($2)) smooth frequency with lines ls 2 title 'Uniform'

set title 'Numbers'
set autoscale x
plot 'results.txt' using 3 with lines ls 1 title '\sigma = 1', \
     'results.txt' using 4 with lines ls 2 title '\sigma = 0.5', \
     'results.txt' using 5 with lines ls 3 title '\sigma = 2'

set xrange [-6:6]
set title 'Distribution'
plot 'results.txt' using (rounded($3)) smooth frequency with lines ls 1 title '\sigma = 1', \
     'results.txt' using (rounded($4)) smooth frequency with lines ls 2 title '\sigma = 0.5', \
     'results.txt' using (rounded($5)) smooth frequency with lines ls 3 title '\sigma = 2'

set title 'Numbers'
set autoscale x
plot 'results.txt' using 6 with lines ls 1 title '\lambda = 1', \
     'results.txt' using 7 with lines ls 2 title '\lambda = 0.5', \
     'results.txt' using 8 with lines ls 3 title '\lambda = 2'

set xrange [0:5]
set title 'Distribution'
plot 'results.txt' using (rounded($6)) smooth frequency with lines ls 1 title '\lambda = 1', \
     'results.txt' using (rounded($7)) smooth frequency with lines ls 2 title '\lambda = 0.5', \
     'results.txt' using (rounded($8)) smooth frequency with lines ls 3 title '\lambda = 2'
