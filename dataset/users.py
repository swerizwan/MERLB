# Define the file name that stores the list of channels
file = 'users.dat'

# Function to add a channel name to the users.dat file
def add(channel: str):
    # Open the file in append mode ('a') to add new data without overwriting
    f = open(file, 'a')
    # Add a new line before writing the channel name
    f.write('\n')
    # Write the channel name to the file
    f.write(channel)
    # Close the file to save changes
    f.close()

# Function to remove a channel name from the users.dat file
def remove(channel: str):
    # Ensure the channel being removed is not "deadfracture" (a protected channel)
    if (channel != "deadfracture"):
        # Open the file in read mode to load all the current entries
        f = open(file, 'r')
        # Initialize an empty list to store updated lines
        lst = []
        # Loop through each line in the file
        for line in f:
            # Replace the channel name with an empty string if found in the line
            line = line.replace(channel, '')
            # Add non-empty lines to the list (avoid empty lines)
            if (line != '\n'):
                lst.append(line)
        # Close the file after reading
        f.close()
        # Open the file in write mode ('w') to overwrite it with updated lines
        f = open(file, 'w')
        # Write each remaining line back to the file
        for line in lst:
            f.write(line)
        # Close the file to save changes
        f.close()

# Function to list all the channels in the users.dat file
def list():
    # Open the file and use a list comprehension to strip trailing newline characters
    with open('users.dat') as f:
        # Create a list of all lines without any extra newlines at the end
        lines = [line.rstrip() for line in f
