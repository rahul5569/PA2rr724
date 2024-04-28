## AWS Spark Wine Quality Prediction Application


The aim of this assignment is to train a machine learning model parallely on ec2 instances for predicting wine quality using the publicly available data and trained model to predict the wine quality. This assignment also uses Docker to create a container image for trained machine learning model to simplify deployments. This is application built using pySpark interface of Python and it is deployed on AWR EMR cluster.

Link to github code –

Link to docker container image –
https://hub.docker.com/repository/docker/rr724/cloudcomputingpa2/general\

Instruction to use:
1.	Create Spark cluster in AWS
- User can create an spark cluster by using EMR console provided by AWS. The following are the steps to create EMR cluster.
- Create Key-Pair for EMR cluster using navigation EC2-> Network & Security -> Key-pairs.
- Use .pem as format. This will download the .pem file. Keep it safe you as you will need it for login to EC2 instances.
- Navigate to Amazon EMR console. Then, navigate to clusters-> create cluster.
- General Configuration -> Cluster Name 
- Software Configuration-> EMR 5.33, do select 'Spark: Spark 2.4.7 on Hadoop 2.10.1 YARN and Zeppelin 0.9.0' option menu.
- Hardware Configuration -> Make instance count as 4
- Security Access -> Provide .pem key created in above step.
- Rest of parameters can be left default.
-	Cluster status should be 'Waiting' on successful cluster creation.

2.	Create s3 bucket for storing code and the data. Upload the code and date to the s3 bucket. The URL should look something like this:  s3://wine-data-rr724/

3. How to train ML model in Spark cluster with 4 ec2 instances in parallel
Now when cluster is ready to accept jobs, submit one you can either use step button to add steps or submit manually.
-	To submit manually, Perform SSH to Master of cluster using below command:
        ssh -i "ec2key.pem" <<User>>@<<Public IPv4 DNS>>
-	On successful login to master , change to root user by running command: sudo su
-	Submit job using following command: spark-submit s3://wine-data-rr724/wine_prediction.py


        
4. Run ML model using Docker
- Install docker 
- Build the image of the docker file using command cmd – docker build -t wine_docker1 .
- You can push this in docker hub repository
- docker push docker push rr724/cloudcomputingpa2:pa2assignment
- Place your testdata file in a folder (let’s call it directory dirA) , which you will mount with docker container and run it below using the below command.

docker run -v /Users/<username>/<path-to-folder>/<appname> /data/csv:/code/data/csv <username>/<user> testdata.csv





