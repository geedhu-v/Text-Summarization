#Install modules
#pip install transformers==2.8.0
#pip install torch == 1.4.0
#pip install sentencepiece

# Importing Modules
import torch
from transformers import T5Tokenizer,T5ForConditionalGeneration,T5Config

#intitialize the pretrained model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')

# Note Article length can be upto 512 words
# Input Text
text = """
Financial research
Overall, financial and investment decisions in organizations require a great deal of investigation and summarization of huge volumes of financial reports and statistics. Individual investors and organizations can use NLP text summarization to identify recurring trends from vast amounts of financial information. This helps stakeholders, financial analysts, investors, and financial advisors to make more informed decisions.

Additionally, NLP text summarization can be used to formulate valuable hypotheses about various financial markets worldwide. This helps financial advisors and researchers to develop better trading strategies that can help companies increase their profits and avoid losses in the long run.

In some cases, NLP text summarization can also be used to generate concise and informative summaries of financial research findings. This way, financial researchers can easily communicate their findings about financial markets and individual securities to other researchers, investors, and the general public.

Media monitoring
Assume you want to know the current state of a given industry from huge amounts of publications and media. However, you don’t have the time to scan through every headline on these publications, let alone read every document related to the industry.

In such cases, text summarization can be used to break down these publications and media into concise and meaningful summaries. With the help of summaries, you’ll be able to stay informed about the current events of a given industry.

Chapters for youtube videos and podcasts
Troubleshooting YouTube videos should always go directly to the point. Viewers are more likely to be frustrated if they have to waste time watching a long YouTube video that doesn’t quickly provide a solution to their problem. The same goes for podcasts. Some listeners want to get to the best bits as soon as possible.

If you create YouTube videos and podcasts, NLP text summarization can be a valuable tool when it comes to generating chapters or sections for your content. This makes it a lot easier for your viewers and listeners to find the specific sections of your content they’re interested in. Breaking up your YouTube videos and podcasts into sections also makes it easier for your viewers and listeners to digest the information you’re trying to present and follow your train of thought.

Email thread summarization
If you run a business and usually receive many emails from clients/customers, you can use a text summarization tool to quickly identify the main points from various conversations. This is particularly helpful, especially if you’re usually busy and want to understand the content in your email inbox without actually reading through hundreds or even thousands of emails.

With the help of a concise and informative summary, you’ll know which particular emails require immediate action and then act accordingly. This will help you stay on top of customer feedback and keep your business running smoothly.
"""
# Preprocess the input text
preprocessed_text = text.strip().replace('\n','')

t5_input_text = 'summarize: ' + preprocessed_text

print(t5_input_text)

print(len(t5_input_text.split()))

tokenized_text = tokenizer.encode(t5_input_text, return_tensors='pt',max_length=512).to(device)

#summary results
summary_ids = model.generate(tokenized_text,min_length=30, max_length=120)
summary = tokenizer.decode(summary_ids[0],skip_special_tokens=True)

print(summary)