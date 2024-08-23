# create_sample_data.py

import os

# Directory to save the sample files
sample_dir = "sample_data"
os.makedirs(sample_dir, exist_ok=True)

# Sample content for the files
samples = {
    "history.txt": "World War II began in 1939 and ended in 1945.",
    "science.txt": "Water is a molecule composed of two hydrogen atoms and one oxygen atom.",
    "technology.txt": "Artificial Intelligence is transforming industries by automating tasks and providing insights.",
    "literature.txt": "Shakespeare's works include tragedies like 'Hamlet' and comedies like 'A Midsummer Night's Dream'.",
}

# Create text files with the sample content
for filename, content in samples.items():
    with open(os.path.join(sample_dir, filename), "w") as file:
        file.write(content)

print(f"Sample data created in the '{sample_dir}' directory.")
