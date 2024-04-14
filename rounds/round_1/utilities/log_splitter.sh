# array of files: sandbox_logs.json, activities.csv, trade_history.json
files[0]=""
files[1]="sandbox_logs.json"
files[2]="activities.csv"
files[3]="trade_history.json"

# set index for which file to write to
index=0
folder_index=0
while [ -d "logs_$folder_index" ]; do
    folder_index=$((folder_index+1))
done
mkdir "logs_$folder_index"

# iterate over lines in the file
while IFS= read -r line; do
    # If the file contains "log" or "History", skip the line and select next file
    if [[ $line == *"log"* ]] || [[ $line == *"History"* ]]; then
        # increment index
        index=$((index+1))
        continue
    fi

    # Else, print the line into the selected file
    echo "$line" >> "logs_$folder_index/${files[index]}"

done < "$1"
