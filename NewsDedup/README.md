Here is a solution to dedup news articles that talk about the same event.
The ipython notebook uses 267 news articles from Jan 23, 2017 (255 valid) as an example.

Highlights:
1. Apply tf-idf on Named Entities extracted from news contents - I'll show how I compute tf-idf from scratch.
2. Use graph to model similarity matrix and combine a title-based similarity model and the aforementioned content model together.

This is an unsupervised task, so I don't have accuracy measure to optimize the model. But spotchecks now and then are satisfying. In addition, I use this script to dedup news feeds for my team (of course with some web sracping and interaction with DB which I eliminated here) and get no complaints so far
