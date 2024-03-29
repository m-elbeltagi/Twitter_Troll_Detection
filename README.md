# Twitter_Troll_Detection
This is the code for a text classification project that aims to distinguish real tweets from "troll" tweets (or rather tweet-farm generated political tweets sponsored by foreign actors). We take a transfer learning approach in general, 3 models were used to compare their performance. (1) A distilled version of a pretrained LLM with fixed weights for encoding the text, and a binary classification head trained on top of it. (2) A pretrained smaller version of a LLM now its weights adjustable by the training process to be finetuned, along with a binary classification head. (3) and finally a new few-shot learning (without prompts) technique called [SetFit](https://huggingface.co/blog/setfit) that dramatically decreases the number of examples needed for training by using contrastive learning. The models were trained using 2 political tweets datasets, one confirmed to be russian-troll tweets, and one containing election-related political tweets, while excluding foreign/ and non-english language tweets.

  
[***The poster won 3rd place at the Carleton Data Day 9.0 poster competition***](https://science.carleton.ca/dataday9/).

![Screenshot](project_poster.png "Project Poster")
