import numpy as np

input = "/Users/rebeccanguyen/Documents/GitHub/H22/AoC/Dec1.txt"
test = []
with open (input) as infile:
    for line in infile:
        elf = line.split("\n\n")
        print(elf)
