#!/bin/bash

# Function to generate a 10-digit random number
generate_random_number() {
    number=""
    for i in {1..10}; do
        digit=$((RANDOM % 10))
        number="${number}${digit}"
    done
    echo "$number"
}

# Arrays of options for each argument
ps=('l' 'h')
# ps=('l')
ms=('large')
pres=(0 1 2 3 4 5 10 15 20)
# pres=(1)
ss=('full')

# Generate a 10-digit random number
ts=$(date +"%m%d%H%M%S")
ts="1105012130"
echo "Timestamp: $ts"
# ts="0121181130"

# Loop from 1 to 10, incrementing by 1
for (( i=1; i<=5; i++ )); do
    # Loop over each combination of arguments
    python A_2.py -ts "$ts-$i" -dp &
done

# Wait for all background processes to finish
wait
