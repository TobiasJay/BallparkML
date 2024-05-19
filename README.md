# Predicting MLB Games

This was originally a class project from a ML class that I have expanded occasionally throughout my second semester. Currently working to improve predictions using an ELO model inspired by video game ranked systems. It is much simpler, or will start out much simpler. I will only compare wins and losses by teams and stats will not be considered.

Next steps could also involve a scraper to get the data during the current season. I don't want to manually copy the data from a website.

Most of main does preprocessing. Models are run at the end at the ====== Train Models ======= section (line 136)

The season, (2023, 2022 or 2021) can be changed on line 124. Simply change the variable season to match the desired date year. Currently the dataset contains only 2021 through 2023.

That's about it. There were three different approaches to the preprocessing and we had to start over each time, but finally, we got it where it needed to be. Adding more columns/features would be an interesting task for another day. Maybe the OBP or the slugging metrics would be better predictors for win rate, like mentioned in Denali's presentation on Moneyball, but that is a task for another day.
