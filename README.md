# task_scheduler.py

## What is it?
code example - schedule optimizer using PuLP

## Install dependencies:
bashpip install pulp pandas

## Run it:
bash$ python task_scheduler.py --operations operations_example.csv --process process1_example.csv process2_example.csv process3_example.csv --output schedule.csv [--print]

## MILP Model (PuLP)
The scheduler formulates a Job-Shop Scheduling Problem using the following contraints:

1. ConstraintDescriptionPrecedence: Each step in a job must start only after the previous step finishes
2. No-overlap: Two tasks sharing the same module cannot run simultaneously (binary ordering variable per pair)
3. Makespan: Objective is to minimize the overall completion time

A Big-M formulation enforces the no-overlap constraints using binary variables: for every pair of tasks on the same module, a binary variable decides which runs first.

## CSV Formats

### Operations catalogue:
module, operation, duration_seconds

The wait operation has an empty module and a duration_seconds of 0 (the actual duration is provided per-job via parameters)

### Process files
module, operation, parameters

Normal steps reference the catalogue by module
Wait steps: module is blank, operation is wait, parameters is duration=<minutes>

### Output DataFrame columns
job_file, job_index, step, module, operation, parameters, duration_seconds, start_time_seconds, end_time_seconds — sorted by start time.
