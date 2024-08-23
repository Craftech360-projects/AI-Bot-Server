# create_ww2_sample_data.py

import os

# Directory to save the sample file
sample_dir = "sample_data"
os.makedirs(sample_dir, exist_ok=True)

# Sample content about World War II
content = """
World War II, also known as the Second World War, was a global war that lasted from 1939 to 1945. 
It involved the vast majority of the world's countries—including all the great powers—forming two opposing military alliances: 
the Allies and the Axis. 

World War II was the deadliest conflict in human history, marked by 70 to 85 million fatalities, 
most of whom were civilians in the Soviet Union and China. 
The conflict saw mass death of civilians, including the Holocaust (in which approximately 6 million Jews were murdered), 
strategic bombing, premeditated death from starvation and disease, and the only use of nuclear weapons in war.

The war in Europe ended with the unconditional surrender of Germany in May 1945, but the war in Asia continued until 
Japan surrendered in September 1945 after the atomic bombings of Hiroshima and Nagasaki. World War II was characterized by significant events 
that shaped the modern world, such as the beginning of the Cold War, the establishment of the United Nations, 
and the acceleration of decolonization movements in Asia and Africa.
"""

# File path for the World War II text file
file_path = os.path.join(sample_dir, "world_war_ii.txt")

# Create the text file with the sample content
with open(file_path, "w") as file:
    file.write(content)

print(f"Sample World War II data created in '{file_path}'")
