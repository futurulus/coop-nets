This is the unfiltered version of the corpus, including incomplete games and games from participants who self-reported as non-native English speakers. The corpus is composed of four csvs:

* `messages.csv` contains every message sent through the chatbox with metadata including `gameid` (the id of the game in which it was sent), `time` (the timestamp), `roundNum` (the round of the game), and `sender` (who sent it: "speaker" or "listener"?).

* `clicks.csv` contains listener responses for every trial with additional metadata including
  * `condition` (the category of trial),
  * HSL coordinates for all three colors,
  * the order they were presented for both players,
  * distances between the colors (measured by delta-e),
  * `outcome` (whether or not they chose correctly)

* `subjectInfo.csv` contains metadata and survey responses for each participant who played the game, including
  * `totalLength` (the time it took for them to play the game)
  * `thinksHuman` (whether they thought they were playing with another human)
  * `confused` (whether they understood the instructions),
  * `comments` (any comments they had at the end),
  * `ratePartner` (how much they liked playing with their partner)
  * `role` (listener or speaker)
  * `nativeEnglish` (whether they are a native english speaker)
  * `score` (out of 50; not collected for all participants)

* `uniqueWorkerIDs.csv` contains anonymized, persistent codes corresponding to the participants who played in each game. This allows us to track repeat players across multiple games.
