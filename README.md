# Predicting MLB Games
Chai-team: Toby, Devarsh, Harsh

This is a class project from a ML class that I have expanded occasionally throughout my second semester. Still doesn't work as well as I'd like it to. I want to try implementing an ELO system this year to see if I can achieve better predictions.

Most of main does preprocessing. Models are run at the end at the ====== Train Models ======= section (line 136)

The season, (2023, 2022 or 2021) can be changed on line 124. Simply change the variable season to match the desired date year. Currently the dataset contains only 2021 through 2023.

That's about it. There were three different approaches to the preprocessing and we had to start over each time, but finally, we got it where it needed to be. Adding more columns/features would be an interesting task for another day. Maybe the OBP or the slugging metrics would be better predictors for win rate, like mentioned in Denali's presentation on Moneyball, but that is a task for another day.
