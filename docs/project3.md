# Categorizing, Clustering, Same Topic articles

## GENERAL IDEA

The general idea is to build a data pipeline that would daily process articles / texts
from specific sources and about specified topics. For this we would scrape those site and / or use services
such as webhose.
Let's assume that we would do this first for tech articles in english from specific sources such
as Techcrunch, Wired and Thenextweb. However, the idea is to build a pipeline that
could be ported to other domains and languages just changing the language processing component or retraining
the engine for a different topic.

The articles would be clustered together with related articles and assigned categories and tags.
Articles discussing the very same topic, possibly from a different slant, would also be recognized.

The retrieved articles descriptions would be made accessible by means of web services. Web services could be integrated
in a web application  where users could manage preferences in their account, for instance saving in what topics
they are interested or what articles they found interesting or uninteresting.

## CATEGORIZING/TAGGING ARTICLES

A most obvious use of this would be to categorize and tag articles from the selected sources and present
/publish only articles regarding specific topics.
When retrieving articles, the topic of interest could be passed as an argument to the web service
or picked by users in their preferences.

We could use the tagging made on same sites to train the categorization and tagging engine (such as Techcrunch), and apply this tagging / categorization to articles from other sites

## CLUSTERING RELATED ARTICLES

A web service could be asked to retrieve articles that are related to a set of other articles (passed
as argument). Moreover, a user in the web application could "like" or "dislike" articles and the system
would then propose articles based on these choices - closer to the articles that he liked, far away from the
articles he disliked

## FINDING ARTICLES DISCUSSING THE SAME TOPIC

Assuming we retrieve articles from several sources, it might be interesting to retrieve articles discussing
the very same topic, but from a separate source, possibily having a different slant.
The engine would return articles that are not only "related", but are actually presenting the same
fact / topic using possibly different words.

## EXTENSION

If time allows, it would be interesting to check the same concept on a different language.
For instance, German, in the same domain (Tech news) , aggregating, categorizing, tagging,
clustering articles from portals such as Chip, Heise and Computerwoche.