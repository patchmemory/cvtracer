#!/bin/bash
# Run this script from the base directory of your local copy of cv-tracer/
# For example:
#  >  cd cv-tracer
#  >  ./scripts/init.sh

echo "Modifying home path for the following script files... "
for file in scripts/*/*
do 
    echo $file
    sed -i "/cvhome=/c\cvhome=\"`pwd`\"" $file
done
