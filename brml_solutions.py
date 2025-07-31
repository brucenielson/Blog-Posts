import os

# Define source directory and output file
source_dir = r"D:\Documents\Books\BRMLtoolkitSolutions"
output_file = os.path.join(source_dir, "combined_output.txt")

with open(output_file, 'w', encoding='utf-8') as outfile:
    for entry in sorted(os.scandir(source_dir), key=lambda e: e.name.lower()):
        if entry.is_file() and entry.name.lower().endswith(".m"):
            try:
                with open(entry.path, 'r', encoding='utf-8') as infile:
                    outfile.write(f"--- {entry.name} ---\n")
                    outfile.write(infile.read())
                    outfile.write("\n" * 5)  # Add 5 newlines between files
            except Exception as e:
                print(f"Could not read {entry.name}: {e}")

print(f"Finished. Combined content is saved to: {output_file}")
