from typing import Set, NamedTuple, List, Tuple, Dict, Iterable
import re
import math
from collections import defaultdict
from dsfs.scratch.machine_learning import split_data # https://github.com/joelgrus/data-science-from-scratch/tree/master/scratch
import csv
 
def tokenize(text: str) -> Set[str]:
    text = text.lower()
    rem_links = re.sub(r'http\S+', '', text) # deletes links
    all_words = re.findall("[a-z0-9]+", rem_links)
    return set(all_words)
   
class Post(NamedTuple):
    text: str
    is_what: int
 
class SentimentClassifier:
    def __init__(self, k: float = 0.333, m: int = 3) -> None: # set the parameter (m) of the minimum number of occurrences of the word
        self.k = k
        self.min_counts = m
 
        self.tokens: Set[str] = set()
        self.token_neg_counts: Dict[str, int] = defaultdict(int)
        self.token_pos_counts: Dict[str, int] = defaultdict(int)
        self.token_neu_counts: Dict[str, int] = defaultdict(int)
        self.all_words: Dict[str, int] = defaultdict(int)
        self.neg_posts = self.pos_posts = self.neu_posts = 0
 
    def train(self, posts: Iterable[Post]) -> None:

        for post in posts:
            if post.is_what == 0:
                self.neg_posts += 1
            elif post.is_what == 4:
                self.pos_posts += 1
            elif post.is_what == 2:
                self.neu_posts += 1
            else:
                break
       
            for token in tokenize(post.text):
                self.all_words[token] += 1
                for _, freq in self.all_words.items():
                    if freq >= self.min_counts:
                        self.tokens.add(token)
                        if post.is_what == 0:
                            self.token_neg_counts[token] += 1
                        elif post.is_what == 4:
                            self.token_pos_counts[token] += 1
                        elif post.is_what == 2:
                            self.token_neu_counts[token] += 1
                        else:
                            break
                    else:
                        break    
           
    def _probabilities(self, token: str) -> Tuple[float, float, float]:
        """Returns P(token | neg), P(token | pos) and P(token | neu)"""
        neg = self.token_neg_counts[token]
        pos = self.token_pos_counts[token]
        neu = self.token_neu_counts[token]
       
        p_token_neg = (neg + self.k) / (self.neg_posts + 3 * self.k)
        p_token_pos = (pos + self.k) / (self.pos_posts + 3 * self.k)
        p_token_neu = (neu + self.k) / (self.neu_posts + 3 * self.k)
       
        return p_token_neg, p_token_pos, p_token_neu
 
    def predict(self, text: str) -> List[float]:
        text_tokens = tokenize(text)
        log_prob_if_neg = log_prob_if_pos = log_prob_if_neu = 0.0
 
        for token in self.tokens:
            prob_if_neg, prob_if_pos, prob_if_neu = self._probabilities(token)
       
            if token in text_tokens:
                log_prob_if_neg += math.log(prob_if_neg)
                log_prob_if_pos += math.log(prob_if_pos)
                log_prob_if_neu += math.log(prob_if_neu)
            else:
                log_prob_if_neg += math.log(1.0 - prob_if_neg)
                log_prob_if_pos += math.log(1.0 - prob_if_pos)
                log_prob_if_neu += math.log(1.0 - prob_if_neu)
           
        prob_if_neg = math.exp(log_prob_if_neg)
        prob_if_pos = math.exp(log_prob_if_pos)
        prob_if_neu = math.exp(log_prob_if_neu)
       
        return [prob_if_neg / (prob_if_neg + prob_if_pos + prob_if_neu),
                prob_if_pos / (prob_if_neg + prob_if_pos + prob_if_neu),
                prob_if_neu / (prob_if_neg + prob_if_pos + prob_if_neu)]

def main():
    data: List[Post] = []
    filename = './testdata3.csv'

    with open(filename, 'r', newline='', encoding='cp437') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for line in csv_reader:
            polarity = int(line[0])
            tweet_text = line[1]
            data.append(Post(tweet_text, polarity))
 
    train_posts, test_posts = split_data(data, 0.75)
   
    model = SentimentClassifier(k=0.333)
    model.train(train_posts)
   
    predictions = [(post, model.predict(post.text))
                   for post in test_posts]

    counter : Dict[str, int] = defaultdict(int)

    for post, prob in predictions:
        if post.is_what == 0:
            if prob[0] == max(prob):
                counter['correct'] += 1
            else:
                counter['wrong'] += 1
        elif post.is_what == 4:
            if prob[1] == max(prob):
                counter['correct'] += 1
            else:
                counter['wrong'] += 1   
        elif post.is_what == 2:
            if prob[2] == max(prob):
                counter['correct'] += 1
            else:
                counter['wrong'] += 1
        else:
            break

    number_of_correct = counter["correct"]
    number_of_wrong = counter["wrong"]

    result = (number_of_correct / (number_of_correct + number_of_wrong)) * 100
    print(result)
   
if __name__ == "__main__": main()
