#!/bin/bash
#SBATCH --mail-user=guedon@et.esiea.fr
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --account=def-glatard
#SBATCH --time=00:05:00
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

module load spark/2.3.0
module load python/3.7

source ~/acc_env/bin/activate

export PYSPARK_PYTHON="/home/tguedon/acc_env/bin/python"
export MKL_NUM_THREADS=1
export SPARK_IDENT_STRING=$SLURM_JOBID
export SPARK_WORKER_DIR=$SLURM_TMPDIR
export SLURM_SPARK_MEM=$(printf "%.0f" $((${SLURM_MEM_PER_NODE} *95/100)))

#start master
start-master.sh
sleep 20


echo "this thing: " $SPARK_LOG_DIR/spark-${SPARK_IDENT_STRING}-org.apache.spark.deploy.master*.out
'
MASTER_URL=$(grep -Po '(?=spark://).*' $SPARK_LOG_DIR/spark-${SPARK_IDENT_STRING}-org.apache.spark.deploy.master*.out)
echo "master url : " ${MASTER_URL}

NWORKERS=$((SLURM_NTASKS - 1))
SPARK_NO_DAEMONIZE=1 srun -n ${NWORKERS} -N ${NWORKERS} --label --output=$SPARK_LOG_DIR/spark-%j-workers.out start-slave.sh -m ${SLURM_SPARK_MEM}M -c ${SLURM_CPUS_PER_TASK} ${MASTER_URL} &
slaves_pid=$!

acc=/home/tguedon/projects/def-glatard/tguedon/accident-prediction-montreal/accidents_montreal.py
road=/home/tguedon/projects/def-glatard/tguedon/accident-prediction-montreal/road_network.py
weather=/home/tguedon/projects/def-glatard/tguedon/accident-prediction-montreal/weather.py
preprocess=/home/tguedon/projects/def-glatard/tguedon/accident-prediction-montreal/preprocess.py
utils=/home/tguedon/projects/def-glatard/tguedon/accident-prediction-montreal/utils.py

srun -n 1 -N 1 spark-submit /home/tguedon/projects/def-glatard/tguedon/accident-prediction-montreal/main.py --master ${MASTER_URL} --py-files ${acc} ${road} ${weather} ${preprocess} ${utils} --executor-memory ${SLURM_SPARK_MEM}M

kill $slaves_pid
'
stop-master.sh
