# CREATING SRT SUBTITLES INTERVALS OF TIMES FROM MUSIC FILES


## MODEL RECOGNIZING HUMAN VOICES IN SONG

MP3 SONG --> SRT FILE

The model produced by the portfolio project should be able to identify continuos portion of songs in which somebody is singing
and produce the approximate intervals of time in form of SRT files,
with placeholder in form of **TEXT{n}** instead of the actual text

An example of output for a song portion would be this -  it means that words
are being spoken between the 32th and the 35th second, and the 35th and 38th second

4
00:00:32,477 --> 00:00:35,020
**TEXT1**

5
00:00:35,020 --> 00:00:38,169
**TEXT2**

The language of the text should be not relevant.
To train this model we would use a database of songs along with their respective SRT files.

Producing the actual text would be out of scope of this step

## POSSIBLE EXTENSION - INTEGRATING WITH THE LYRICS

There is much higher availability of song lyrics than of SRT files. A possible extension
would be to take the MP3 song file and the song lyrics as input, and produce the SRT FILE as output

MP3 SONG  ----------\
                     ----> SRT FILE
(RAW) LYRICS -------/

Note that it would be not enough to match lyrics with placeholders in SRT files sequentially,
because lyrics often do not contain the text lines in the correct order
or text lines are repeated several times during the song.

This could be accomplished interfacing with a speech recognition API and
then trying to find the most likely match among the lyrics song lines.

## POSSIBLE EXTENSION - INTEGRATE WITH YOUTUBE

Instead of producing the SRT File from a downloaded song, it might be possible to produce the SRT file
from a url pointing to a music video and a lyrics file or even a URL pointing to a page containing the lyrics


YOUTUBE URL--------\
                    ----> SRT FILE
(RAW) LYRICS-------/

## What are SRT Files

SRT Files are used to subtitle videos. The SRT format contains a list of intervals of time
when words are being spoken or sung, and the actual text spoken.

For instance below you can see the words that in the associated videos/song are being spoken
between the time intervals 00:00:32,477 --> 00:00:35,020 and 00:00:35,020 --> 00:00:38,169


4
00:00:32,477 --> 00:00:35,020
Qui dit étude dit travail,

5
00:00:35,020 --> 00:00:38,169
Qui dit taf te dit les thunes,
