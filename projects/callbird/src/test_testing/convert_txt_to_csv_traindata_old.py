# list all files in the current directory
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
    chunk_files = {}

    # Read each .txt file and append to the DataFrame
    for file in files:
        if file.endswith('.txt'):
            first_bracket = file.find('_')
            chunk_name = file[:first_bracket] if first_bracket != -1 else file
            chunk_counts[chunk_name] = chunk_counts.get(chunk_name, 0) + 1
            chunk_files[chunk_name] = chunk_files.get(chunk_name, []) + [file]

    # species_dict = {}

    global_initial_line = ""
    for chunk_name in chunk_files:
        # chunk_count = chunk_counts[chunk_name]
        # if chunk_count < 10:
        #     print(f"Warning: Chunk {chunk_name} has less than 10 files, skipping.")
        #     continue

        initial_line = ""
        output_file = os.path.join(target_directory, f"{chunk_name}.csv")
        with open(output_file, 'w') as csv_file:
            for file in chunk_files[chunk_name]:
                with open(os.path.join(directory, file), 'r') as txt_file:
                    lines = txt_file.readlines()
                    if len(lines) != 2:
                        print(f"Warning: {file} does not have exactly 2 lines, skipping.")
                        continue
                    if not initial_line:
                        initial_line = lines[0].replace('\t', ';')
                        csv_file.write(initial_line.replace('\n', '') + ";ebird_code\n")

                        if not global_initial_line:
                            global_initial_line = initial_line
                        elif initial_line != global_initial_line:
                            print(f"Warning: Chunk {chunk_name} has a different header, skipping.")
                            continue
                    elif initial_line != lines[0].replace('\t', ';'):
                        print(f"Warning: {file} has a different header, skipping.")
                        continue
                    data_line = lines[1].replace('\t', ';')
                    species = data_line.split(';')[6]

                    if species in testSpeciesToEbirdCodeDict:
                        csv_file.write(data_line.replace('\n', '') + f";{testSpeciesToEbirdCodeDict[species]}\n")
                        # csv_file.write(data_line)
                    # sub_species = data_line.split(';')[7]
                    # if species not in species_dict:
                    #     species_dict[species] = set()

                    # species_dict[species].add(sub_species)


    # print chunk_dict in rows from biggest to smallest
    sorted_chunks = sorted(chunk_counts.items(), key=lambda x: x[1], reverse=True)
    # print total chunks
    print(f"Total chunks: {len(sorted_chunks)}")
    for chunk, count in sorted_chunks:
        print(f"{chunk}: {count}")

"""
    # write species_dict in file biggest to smallest
    species_name_output = os.path.join("/workspace/projects", "train_species_dict.txt")
    
    sorted_species = sorted(species_dict.items(), key=lambda x: len(x[1]), reverse=True)
    with open(species_name_output, 'w') as species_file:
        for species, sub_species in sorted_species:
            species_file.write(f"{species}: {len(sub_species)}\n")
            for sub in sub_species:
                species_file.write(f"> {sub}\n")
            species_file.write("\n")
"""
            
convert_txt_to_csv("/workspace/oekofor/trainset/txtlabels", "/workspace/oekofor/trainset/csvlabels")

# convert_txt_to_csv("./txt_test")
# convert_txt_to_csv("C:/Users/Morit/Downloads/bird_set_test_data/txtlabels/txtlabels")