#!/bin/bash


for file in JSON/*;
	do python3 main.py $file;
done;
