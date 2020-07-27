# Jooble test task
## Brief description
This repository contains code which is the implementation of the task.
The code may be divided in three parts:
  1) Script ```generate_file.py``` generates ```test_proc.tsv``` for given data from ```/data``` folder.
  2) Modules ```local_statistics.py``` and ```global_statistics.py``` contain classes that calculate statistics.
  3) Folder ```unittest``` contains data for testing along with test classes.

To generate ```test_proc.tsv``` invoke ```python generate_file.py``` in cmd.

## Details of implementation
All statistics that should have been calculated can be divided in 2 groups:
  1) First group contains statistics that need entire dataset to be estimated. Such statistics are ```mean``` 
     or ```std```. For calculation of these statistics I created abstract class ```GlobalStatistic```. Each subclass 
     implements some statistic that should be calculated. In our case - ```mean``` and ```std```. These classes can 
     estimate iteratively, which may be helpful in case of large datasets. So we can pass chunks of data one after 
     another and calculate statistic's estimation in many steps - some kind of learning. Also these classes have methods 
     to iterate over a large file and do the job.
  2) Second group contains statistics that need only one row for evaluation. Such statistic is ```max_index```. 
     For calculation of these statistics I created abstract class ```LocalStatistic```. Each subclass implements 
     some statistic that should be calculated. In our case - ```max_index```, ```max_abs_mean_diff```, ```z_score```. 
     These classes receive dataframe, calculate statistic for each row and add a column with this statistic to the dataframe.
 
 I took into consideration all the factors mentioned in the task:
 1) __The ability to add new features in the future.__ My classes work with different features with different dimensions.
    Each ```GlobalStatistic``` subclass has dictionary that holds value of statistic for each feature. Also vectorization
    is done with consideration of fact that different features may have different dimensions.
 2) __Large size of input files.__ All the calculations are performed iteratively, so large data may be divided and passed to 
 classes in smaller chunks. So no matter, how much rows are in the files, we will be able to gain the result. 
 3) __The ability to add new statistics in the future.__ Each new statistic may be created as implementation of appropriate 
 abstraction (```GlobalStatistic``` or ```LocalStatistic```).
 
 ## Ways to improve
 During development I was thinking about improvements that may speed up performance of my solution. In case of large 
 data it may be important. 
  1) First of all python is known for its slow I/O operations. So it may be beneficial to use 
     separate thread for reading chunks of data into some queue.
  2) In my implementation ```mean``` and ```std``` are calculated in two runs. There is an iterative formula for calculating
     ```mean``` and ```std``` simultaneously. This implementation may help speed up calculations as well.
 
