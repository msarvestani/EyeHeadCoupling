To run Erin's eye head coupling analysis:



1\. run EyeHeadCouplingAnalysis.m (this file batch processes all the data folders, outputs figures for individual sessions (i.e. videos showing saccade/imu, plots showing eye tracking across the session, etc.) and saves stats about each saccade from all the sessions in saccade\_imu\_population\_table.mat)

* inputs: file\_database.mat that contains the paths to all the sessions you want to run (file\_database\_master contains paths to ALL data files, file\_database only contains files that passed my data quality criteria)
* outputs (in whatever folder you set as save\_directory)
* &nbsp;	figures (if you set any of the plotting\_params as 1)
* &nbsp;	saccade\_imu\_population\_table.mat (see description of contents below)
* &nbsp;	file\_database\_stats.mat/.csv (contains statistics about each session like duration, number of saccades, tracking quality, etc - useful for keeping track of data quality)\\
*  	dlc\_likelihood\_table.mat (contains average likelihood for each of the 12 points for each eye, can be useful in determining whether tracking was good for each session)

2\. then run EyeHeadCouplingAnalysis\_populationPlots.m (this file takes in the saccade\_imu\_population\_table with all detected saccades and computes population statistics/makes population level plots

* inputs: saccade\_imu\_population\_table.mat (from running EyeHeadCouplingAnalysis.m)
* outputs: plots with population level analysis of all the saccades





criteria I used to remove bad data:

* imu all 0s/looks weird
* % good tracking <90% or mean dlc likelihood for both eyes <0.9
* <10 min recording



there are many parameters related to the eye tracking that may need to be updated based on improved model performance

* parameters need to be updated based on model performance
* 



to add new files to the analysis:

1. first need to analyze videos in DLC with the best current model to get .csvs with x/y position of each point on the eye across time
2. additionally, if you want to output saccade/imu videos for manual verification of saccades: need to create videos in DLC
3. add the paths into the file\_database.mat file





saccade\_imu\_population\_table

* each row is a single saccade
* columns for session id, stats like magnitude, direction, timestamp, etc - also has full eye trace and imu signal for 1 sec before/after saccade
* contains all detected saccades from all sessions that were run
