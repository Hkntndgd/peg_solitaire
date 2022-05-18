For information you can visit wiki :https://en.wikipedia.org/wiki/Peg_solitaire

Main steps:
	First Step:
	Pawns are represented by 1, free places by 0 on the table
	Play 500 times randomly
	Construct a score card for each coordinate on the table.
	It is simply sum of last stage of 500 tables
	Score of a motion is calculated by adding 1's score and subtract 0's score;
simply because you are eliminating 1's which is desired but add 1 in 0's position.

	Second Step:
	Identify primary axes and secondary axes in both direction: row wise, column
wise: For every axe calculate average score in all possible direction( for row wise
from right to left or from left to right; for column wise from top to down or from
down to top) If an axe has an average in one direction greater than the average of all
axes it is considered as a underlying axe. Visit weak_link(ones) procedure.

	Third Step:
	As long as there are possible motions along underlying axes, you are forced to
play among them. 
	Meanwhile according to a random variable output named orienter, you might play 
best score motion.
	Visit simulate_move_randomly_learned(ones,prob,underlying_axe_row_wise,
underlying_axe_column_wise ) procedure

	Fourth Step:
	Play until success 30 times with different probabilities: 0.4 to 0.8 delta=0.05
	Results are presented in probability_optimization_boxplot.png and probability_
optimization.png

	Fifth Step:
	Confirm probability = 0.65
	Histogram is on confirmation_over_100_plays.png 
	Mean is around 1200 play.

	Finaly:
	Create a GIF to illustrate a succesfull play.
	
	Conclusion:
	After playing 500 times to learn, with a probability parameter of 0.65 you will get 
	a solution in 1228 plays on average and in approximatively 2933 plays with 90% of 
	confidence interval.
	A density histogram of 100 successfull plays results is compared with 1000 randomly 
generated from gamma distribution RV (shape=1,scale=mean of 100 plays) to show the similitude.
	