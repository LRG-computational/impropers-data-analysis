#!/bin/bash

# Define the output file name
output_file="trajectory.xyz"

# Check if the output file already exists and remove it to start fresh
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

# Loop over the range of Phi and Theta values
for phi in {0..60}; do
    for theta in {0..360}; do
        # Construct the file name
        file="Dimethyl_Naphthalene_Dicarboximide_Phi_${phi}_Theta_${theta}_Thiophene.xyz"

        # Check if the file exists and append it to the output file
        if [ -f "$file" ]; then
            cat "$file" >> "$output_file"
            # Optional: Add a line break between files if your XYZ format requires it
            # echo "" >> "$output_file"
        fi
    done
done

echo "Concatenation complete. Output in $output_file"
