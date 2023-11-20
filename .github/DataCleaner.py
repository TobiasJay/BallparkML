
# Specify the path to the text file
# text copied from https://www.baseball-reference.com/leagues/majors/2023-schedule.shtml
file_path = "../data/RegularSeasonSchedule.txt"

# Open the file in read mode
with open(file_path, "r") as file:
    # Read the contents of the file
    file_contents = file.read()

# Print the contents of the file
print(file_contents)
