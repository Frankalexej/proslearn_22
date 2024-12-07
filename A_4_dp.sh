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

# Generate a 10-digit random number
ts=$(date +"%m%d%H%M%S")
ts="1207190600"
echo "Timestamp: $ts"
# ts="0121181130"

# Loop from 1 to 10, incrementing by 1
for (( i=1; i<=1; i++ )); do
    # Loop over each combination of arguments
    python A_4.py -ts "$ts-$i" -dp &
done

# Wait for all background processes to finish
wait
