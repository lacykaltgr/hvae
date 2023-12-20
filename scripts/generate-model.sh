#!/bin/bash

generate_model_file() {
    file_name=$1
    template="models/template.py"

    if [ -e "$file_name" ]; then
        echo "Error: File '$file_name' already exists."
        return 1
    fi

    if [ ! -f "$template" ]; then
        echo "Error: Template file '$template' does not exist."
        return 1
    fi

    cp "$template" "$file_name"
    echo "Successfully created '$file_name'."
}

# Call the function to create the new file
generate_model_file "$@"
