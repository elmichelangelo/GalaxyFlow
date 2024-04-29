#!/bin/bash

for i in {1..100}
do
   python run_gaNdalF.py -cf LMU.cfg --spatial $i
done