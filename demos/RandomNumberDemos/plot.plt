# http://psy.swansea.ac.uk/staff/carter/gnuplot/gnuplot_frequency.htm

reset

set datafile separator ";"

set term pngcairo size 1128,800 enhanced
set output 'distribution.png'

set multiplot layout 2,2 title "Distribution\n"

set style line 1 linecolor  rgb "blue"
set style line 2 linecolor  rgb "green"
set style line 3 linecolor  rgb "red"

bin_width = 0.05;
bin_number(x) = floor(x / bin_width)
rounded(x) = bin_width * bin_number(x)

set size 1,0.5

set title 'Uniform'
plot 'results.txt' using (rounded($1)) smooth frequency with lines ls 1 title 'System.Random', \
     'results.txt' using (rounded($2)) smooth frequency with lines ls 2 title 'Uniform'

set multiplot next # we want to skip the second (upright position)
set size 0.5,0.5

# Add a vertical dotted line at x=0 to show centre (mean) of distribution.
set yzeroaxis

set xrange [-6:6]
set title 'NormalDistributed'
plot 'results.txt' using (rounded($3)) smooth frequency with lines ls 1 title '\sigma = 1', \
     'results.txt' using (rounded($4)) smooth frequency with lines ls 2 title '\sigma = 0.5', \
     'results.txt' using (rounded($5)) smooth frequency with lines ls 3 title '\sigma = 2'


set xrange [0:5]
set title 'ExponentialDistributed'
plot 'results.txt' using (rounded($6)) smooth frequency with lines ls 1 title '\lambda = 1', \
     'results.txt' using (rounded($7)) smooth frequency with lines ls 2 title '\lambda = 0.5', \
     'results.txt' using (rounded($8)) smooth frequency with lines ls 3 title '\lambda = 2'
