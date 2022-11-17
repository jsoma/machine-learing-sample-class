Hi! I'm Soma, I run the data program and this is a big fake class.

The strangest thing is there are a million of you, our classes are never more than low 20's people.

But there's one big benefit to this, and that is I can probably convince you to participate a little bit in the chat. So yes, you can throw your questions in there, but most importantly when I say something like "are we ready for this???" you can all type "yes!!!!" in the chat.

So let's try it: "are we ready???"

Cool, let's do it.

Today we're going to cover something that's actually incredibly boring, even though it's about investigative journalism. But that's mostly what computers are good at, is the boring stuff, so you can focus on the more important stuff.

The very very human way of explaining what we're going to do today is this: sorting through documents. I like to think of almost task that you do, there's a spectrum that you can use to talk about it. On one side you have the real broad human-being way of thinking about it that probably everyone can understand:

"sort through some documents"

and on the other side you have - potentially - the incredibly specific technical description that is almost every single one of our implementation details:

"train a binary classifier model on the tokenized 'Review' column using a huggingface transformer (uncased dilbert)"

And then in between you have something that's a little bit of a combination of the two:

"have a machine classify each document as interesting or not"

One of the goals of the classes I teach at Columbia are when you're presented with a problem, you can be told a more human way of thinking about it, and then you can translate that into the technical approach. Or if you're stuck, I can tell you the technical way of doing it, and your head won't explode, and then you'll go implement it.

Our goal today is for you to understand this sentence: "train a binary classification model on the tokenized 'Review' column using a huggingface transformer (uncased dilbert)"

Can we accomplish this? I don't know, we'll find out!

=====

We're going to begin with something not nearly as exciting as investigative journalism. We're going to start with something called SENTIMENT ANALYSIS.

https://investigate.ai/investigating-sentiment-analysis/comparing-sentiment-analysis-tools/

Sentiment analysis is a technique where you can take a short piece of text, feed it into a machine, and the machine will tell you whether it's positive or negative. You can use this to see whether people writing reviews on Amazon like a product, or whether people tweeting about a movie like it, or whether people writing about a politician like them.

The best part sentiment analysis is that you don't really need to know how to program to use it. You just type like three lines and you're done.

Let's do it. We're going to be using one called something or other.

Here's a sentence. Is it positive or negative?

Great, let's see.

*Run sentiment analysis on the sentence*

Okay, another sentence. Positive or negative?

*Run sentiment analysis on the sentence*

Great, we're incredibly accomplished. Super technical, incredible people. And the computer agrees with us.

How does this work? It doesn't matter right now. All we need to know at this moment is that it takes a piece of text and puts it into a category - positive or negative.

You could also say it CLASSIFIES a piece of text. Maybe you could call it a CLASSIFIER.

"What kind of classifier?" You might shout, from the back of the room.

Well, it's only classifying in two directions, right? Positive or negative? Things that only have two settings - off or on, positive or negative, zero or one - that's a BINARY setting. Binary is just, zero or one, off or on, positive or negative.

So if we have something that classifies things into two categories, that's a BINARY CLASSIFIER or BINARY CLASSIFICATION MODEL. The "model" word on the end just means, "a thing that does binary classification."

"train a XbinaryXclassificationXmodelX on the tokenized 'Review' column using a huggingface transformer (uncased dilbert)"

An interesting thing about sentiment analysis is that there's not just one sentiment analysis tool that knows everything, there are like a million of them, and they don't agree with each other.

Let's look at a couple more.

bam

bam

They all give different responses when confronted with the same sentences. Weird, right? But they're all built differently.

For example, I might make a list of all of the words I can think of. Dog. Messy. Smile. Ruined. And then for each of the words I write down how positive or negative it is.

- Dog: kinda positive
- Messy: kinda negative
- Smile: very positive
- Ruined: very negative

And then when I have a sentence, I just add up all of the words in the sentence and see the total is positive or negative.

That's a really old fashioned way of doing it. The problem is I need to make a list of words, and then score each of them, which is just too much work. I don't want to do that.

A slightly more modern way of doing it is taking a bunch of sentences, and each sentence is either positive or negative. And then I take each word in the sentence, and I count how many times it appears in a positive sentence, and how many times it appears in a negative sentence. And then I can use that to score each word.

"Smile" appears in a lot of positive sentences, so it's probably positive. "Ruined" appears in a lot of negative sentences, so it's probably negative.

The problem again is I don't want to do this manually. Where are you going to find a bunch of sentences or short pieces of text that are already marked as being positive or negative?

Movie reviews, Amazon product reviews, anything where people are expressing an opinion online and marking it as I LOVED IT or I HATED IT are a good source of data. 5-star review, great, it's positive. 1-star review, great, it's negative.

These sentiment analysis models all have different sources of data. Some learned their words from Amazon product reviews, some from movie reviews, some from people manually scoring words. This process of teaching what the words are and whether they're positive or negative is called TRAINING YOUR MODEL.

You don't care about how it's done just now, just know that training consists of things like, picking the words you're going to use, picking the threshold as what counts as "positive," stuff like that. Lot of human being decisions going into a technical thing

A lot of times sentiment analysis is used for things like "Elon Musk bought Twitter, are people happy about it or not?" And you'd download a lot of tweets and run them through a sentiment analysis model and see what the average is. It's very easy to do using those tools, but now that we know a little bit more we might be able to think about some shortcomings of it.

Is a tweet a movie review? Is a tweet a product review? Are the words that someoen uses when they're saying they don't like a movie the same as when they're tweeting that they don't like something? So I guess the question is, does it make sense to have a model trained on movie reviews or product reviews evaluating tweets?

Maybe not. But it's really really easy to do, and unless you looked under the hood to see how the sentiment analysis tools were trained, you might just say "oh, we got a number back, we're probably good to go."

We could have a nice big class discussion about this, and when we can and can't use certain kinds of sentiment analysis trained on this or that, but we can't!!!! We don't have time!!! We can just be impressed we understand some more words, and now we're going to move on.

"XtrainXaXbinaryXclassificationXmodelX on the tokenized 'Review' column using a huggingface transformer (uncased dilbert)"

Okay, let's move up the food chain. So far we've been using other peoples' sentiment analysis tools. What if we didn't agree with the way that the existing tools were trained, or we wanted to train it on a different set of data? What if we wanted to make our own?

Well, we need some data. Luckily we have some data. It's a bunch of tweets that are either marked as positive or negative:

http://help.sentiment140.com/home
https://investigate.ai/investigating-sentiment-analysis/designing-your-own-sentiment-analysis-tool/

here it is

Now where did this data come from? Well, someone downloaded all of the tweets that had a smiley face in it and said they were positive and all of the frowny face tweets and said they were negative. Which, I guess, maybe makes sense overall? But also seems kind of silly.

But we're going to use it anyway. We're going to use it to train our own sentiment analysis tool.

The first step is taking all these tweets and counting all the words inside. That way we can figure out whether "smile" shows up in positive tweets or negative tweets. So let's do it, we just run some code and it counts them.

Look at that! Every row is a sentence and every column is a word. And the number in the cell is how many times that word appears in that sentence.

Now we can understand this pretty well, right? We just separated the sentence into words and counted them. Because we're all computer people now, we want to use fancier words to describe it.

"we separated the words" = "tokenization"
"we counted them" = "vectorization"

You might say "it's pretty dumb that tokenization just means separating words" but it get pretty crazy pretty fast. If we hopped to Chinese for example, we don't have spaces between words, so we need some sort of magic other rule to split them apart. Even something like the word "can't" in English - is it a word, is it "ca" and "n't" is it "can" and "'t"? It gets weird.

Anyway, we knocked out a couple other words:

"XtrainXaXbinaryXclassificationXmodelXonXtheXtokenizedX'Review'XcolumnX using a huggingface transformer (uncased dilbert)"

"but soma!" you scream. "we don't have a Review column!"

Yes. That's true. We don't have a Review column, we have a column about tweets, and that's because we still need to do investigative journalism!

Finding out whether a tweet is positive or negative is not actually investigative journalism, I am sorry. Should we do investigative journalism instead?

Well, okay, let's take this example from the Washington Post. They investigated a bunch of app store reviews.

https://www.washingtonpost.com/technology/2019/11/22/apple-says-its-app-store-is-safe-trusted-place-we-found-reports-unwanted-sexual-behavior-six-apps-some-targeting-minors/

They found that there were a bunch of apps that were being used to send inappropriate messages to minors, and they found it out by reading reviews on the app store.

The problem is that there were a lot of reviews on the app store, and they didn't want to read all of them. They needed to call sources and write stories and other like fancy journalism stuff, they didn't have time to read all of the reviews. They didn't have an army of interns to do it for them. So instead: they trained a binary classification model that, instead of looking at positive vs negative, it looked for "this is about inappropriate behavior" vs "this is not about inappropriate behavior."

They manually went through and tagged a series of reviews as being about inappropriate behavior or not, and then they trained a model using that data. So yes, they had to read some of them, but after a certain point they could just say "hey computer, find more reviews that look like this."

And it did.

We're still missing some of these words in our sentence, though:

"XtrainXaXbinaryXclassificationXmodelXonXtheXtokenizedX'Review'XcolumnX using a huggingface transformer (uncased dilbert)"

It turns out that even if we read a few reviews and manually tag them, it still isn't really as smart as it could be. Like if I say cats are positive, what about kittens? You and I know that cats and kittens are similar, but if the word "kitten" didn't show up in the data we trained our model on, it won't know that.

In the case of Washington Post, maybe someone in the training data said they talked to someone who was "gross." But then when we're trying to use the model, trying to classify reviews, someone might say people were "icky" or "disgusting" instead, and if we haven't seen those words before, the model won't know they're similar to "gross."

This is where something called pre-trained models come in. Large organizations like Facebook or Google or things like that made models that didn't just look at positive or negative tweets, they just just read everything on Wikipedia, read a ton of stuff on the internet, and started to learn what words mean. That cats and kittens are similar, that you can eat strawberries and bread but can't eat a car, just little concepts we take for granted that are more complicated than just "count some words".

There are all kinds of different models that exist out there, trained for all sorts of different tasks. Chat bots or language translation or whatever. They have names like ELMo or DilBERT or GPT-3 and they're way more powerful than us just counting words.

So we're going to use one of those. It already knows about the English language, we just need to show it examples of app reviews that are about inappropriate conduct and ones that aren't. If we say "gross" means we probably want to read the review, it will understand that "icky" and "disgusting" probably mean that, too.

*Do it*

And there we go! Works great!

https://investigate.ai/wapo-app-reviews/predict-reviews/

It can go beyond that, though. If a language model knows English, why not German or Spanish or Chinese? Turns out it works just as well in other languages, or even across languages:

https://qz.com/1786896/ai-for-investigations-sorting-through-the-luanda-leaks
https://twitter.com/jeremybmerrill/status/1218979999878909952
https://investigate.ai/text-analysis/comparing-documents-in-different-languages/

And classifiers know more than just language. They can also know about images, or audio, or video. For example, long long ago, a reporter at BuzzFeed used a classifier to predict whether a plane was a surveillance plane or a passenger plane: https://investigate.ai/buzzfeed-spy-planes/

Any honestly this is just a tiny tiny tiny tip of what machine learning and AI are capable of doing.

One of my favorite things that I taught my students a few weeks ago is an AI tool called GitHub Copilot. When you're writing software or doing data analysis, Copilot predicts the code you're going to write and suggests it for you. So you don't necessarily have to remember every single little thing about your code, you can just write a few words and it will fill in the rest.

It also does a good job just writing text in general. For example, I was writing this class out as a script. It helps me plan what we're going to cover, even if I'm not directly reading it, it just works better for me than planning through just a bunch of bullet points.

And so then we get to the end, maybe I don't even get to decide what we talk about, I just let Copilot take over. 