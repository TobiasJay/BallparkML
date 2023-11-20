import re
from datetime import datetime


def parse_game(game):
    # Pittsburgh Pirates (4) @ Boston Red Sox (1)
    # input looks like the above string

    # Output gives the home team, away team, and the score


    # Define a regular expression pattern
    pattern = re.compile(r'(\D+)\s*\((\d+)\)\s*@\s*(\D+)\s*\((\d+)\)')

    # Use the pattern to extract team names and scores
    match = pattern.match(game)

    if match:
        away_name, away_score, home_name, home_score = match.groups()
    else:
        print("No match found/error in input")

    return away_name, away_score, home_name, home_score

def parse_date(date):
    # Your input string for Sunday, April 2, 2023

    # Parse the date using datetime
    parsed_date = datetime.strptime(date, "%A, %B %d, %Y")

    # Format the date in MM/DD/YYYY format
    formatted_date = parsed_date.strftime("%m/%d/%Y")

    # Print the formatted date
    return formatted_date

def main():
            
    # Specify the path to the text file
    # text copied from https://www.baseball-reference.com/leagues/majors/2023-schedule.shtml

    file_path = "data/TestScores.txt"


    # Open the file in read mode
    with open(file_path, "r") as file:
        # Read the contents of the file
        file_contents = file.read()

    # Now we want to change this long string format into a table of data
    parse_game("Pittsburgh Pirates (4) @ Boston Red Sox (1)")
    # First we split by days
    for dailyContents in file_contents.split("»"):
        dailyContents = dailyContents.strip()
        linecount = 0
        formatted_date = ""
        for game in dailyContents.split("Boxscore"):
            if linecount == 0:
                lines = game.split("\n")
                lines = lines[2:] # Remove the first line
                date = lines[0]
                firstgame = lines[1]
                formatted_date = parse_date(date)
                #parse_game(firstgame)
                # save date and firstgame
            else:
                game = game.strip()
                
                #needs further cleaning
                #parse_game(game)
                # save game

            linecount += 1



if __name__ == '__main__':
    main()