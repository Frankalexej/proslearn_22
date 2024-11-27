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
ps=('h')
ms=('twoconvCNN')
pres=(10 20 30 40 50 60 70 80 90 100 150)
# pres=(1)
ss=('full')

# Generate a 10-digit random number
ts=$(date +"%m%d%H%M%S")
ts="1126125730"
echo "Timestamp: $ts"
# ts="0121181130"

for (( i=1; i<=5; i++ )); do
    # Loop over each combination of arguments
    echo "$i"
    for p in "${ps[@]}"; do
        for m in "${ms[@]}"; do
            for s in "${ss[@]}"; do
                for pre in "${pres[@]}"; do
                    # Randomly select a GPU between 0 and 8
                    gpu=$((RANDOM % 9))
                    post=$((200 - pre))

                    # Run the Python script with the current combination of arguments in the background
                    python A_3.py -ts "$ts-$i" -p "$p" -m "$m" -s "$s" -gpu "$gpu" -pree "$pre" -poste "$post" &
                done
            done
        done
    done
    # Wait for all background processes to complete before moving to the next iteration
    wait
done

# # Loop from 1 to 10, incrementing by 1
# for (( i=2; i<=5; i++ )); do
#     # Loop over each combination of arguments
#     for p in "${ps[@]}"; do
#         for m in "${ms[@]}"; do
#             for s in "${ss[@]}"; do
#                 for pre in "${pres[@]}"; do
#                     # Randomly select a GPU between 0 and 8
#                     gpu=$((RANDOM % 9))
#                     post=$((200 - pre))

#                     # Run the Python script with the current combination of arguments in the background
#                     python A_2.py -ts "$ts-$i" -p "$p" -m "$m" -s "$s" -gpu "$gpu" -pree "$pre" -poste "$post" &
#                 done
#             done
#         done
#     done
# done

# # Wait for all background processes to finish
# wait
