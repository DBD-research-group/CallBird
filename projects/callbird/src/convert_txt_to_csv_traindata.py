import os

def convert_txt_to_csv(directory, target_directory="csv_test"):
    testSpeciesToEbirdCodeDict = {}

    with open(os.path.join(directory, "/workspace/projects/speciesEbirdCodeListFlat.txt"), 'r') as speciesToEbirdFile:
        for line in speciesToEbirdFile:
            parts = line.strip().split(':')
            if len(parts) != 2:
                print(f"Warning: Line '{line.strip()}' in speciesEbirdCodeListFlat.txt does not have exactly 2 parts, skipping.")
                continue
            # trim and add to the dictionary
            testSpeciesToEbirdCodeDict[parts[0].strip()] = parts[1].strip()

    # List all files in the directory
    files = os.listdir(directory)

    chunk_counts = {}
    chunk_data = {}

    file_index = 0
    files_count = len(files)
    for file in files:
        file_index += 1
        if file.endswith('.txt'):
            with open(os.path.join(directory, file), 'r') as txt_file:
                lines = txt_file.readlines()
                
                if len(lines) != 2:
                    print(f"Warning: {file} does not have exactly 2 lines, skipping.")
                    continue
                
                header_line = lines[0].replace('\t', ';').replace('\n', '')
                data_line = lines[1].replace('\t', ';').replace('\n', '')
                species = data_line.split(';')[6]

                if species not in testSpeciesToEbirdCodeDict:
                    # print(f"Species '{species}' not in test dataset, skipping file {file}.")
                    continue

                chunk_counts[species] = chunk_counts.get(species, 0) + 1
                
                if species in chunk_data and chunk_data[species][0] != header_line:
                    print(f"Warning: {file} has a different header, skipping.")
                    continue

                chunk_data[species] = chunk_data.get(species, [header_line]) + [data_line]
        print(f"Processed {file_index}/{files_count} files.")

    global_initial_line = ""
    for species in chunk_data:
        # chunk_count = chunk_counts[chunk_name]
        # if chunk_count < 10:
        #     print(f"Warning: Chunk {chunk_name} has less than 10 files, skipping.")
        #     continue

        output_file = os.path.join(target_directory, f"{species}.csv")
        with open(output_file, 'w') as csv_file:
            header_line = chunk_data[species][0]
            csv_file.write(header_line + ";ebird_code\n")
            
            if not global_initial_line:
                global_initial_line = header_line
            elif global_initial_line != header_line:
                print(f"Warning: {file} has a different header, skipping.")
                continue

            for file in chunk_data[species][1:]:  # Skip the header line
                csv_file.write(file + f";{testSpeciesToEbirdCodeDict[species]}\n")

    # print chunk_dict in rows from biggest to smallest
    sorted_chunks = sorted(chunk_counts.items(), key=lambda x: x[1], reverse=True)
    # print total chunks
    print(f"Total chunks: {len(sorted_chunks)}")
    for chunk, count in sorted_chunks:
        print(f"{chunk}: {count}")
            
convert_txt_to_csv("/workspace/oekofor/trainset/txtlabels", "/workspace/oekofor/trainset/csvlabels")