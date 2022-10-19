# Based on the Naive Bayes Classifier from Joel Grus' Data Science from Scratch.

from typing import Set, NamedTuple, List, Tuple, Dict, Iterable
import re
import math
from collections import defaultdict, Counter
import random
from scratch.machine_learning import split_data # https://github.com/joelgrus/data-science-from-scratch/tree/master/scratch

def tokenize(text: str) -> Set[str]:
    text = text.lower()
    all_words = re.findall("[a-z0-9]+", text)
    return set(all_words)
    
class Post(NamedTuple):
    text: str
    is_neg: bool

class SentimentClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k

        self.tokens: Set[str] = set()
        self.token_neg_counts: Dict[str, int] = defaultdict(int)
        self.token_pos_counts: Dict[str, int] = defaultdict(int)
        self.neg_posts = self.pos_posts = 0

    def train(self, posts: Iterable[Post]) -> None:
        for post in posts:
            if post.is_neg:
                self.neg_posts += 1
            else:
                self.pos_posts += 1
        
            for token in tokenize(post.text):
                self.tokens.add(token)
                if post.is_neg:
                    self.token_neg_counts[token] += 1
                else:
                    self.token_pos_counts[token] += 1

    def _probabilities(self, token: str) -> Tuple[float, float]:
        """Returns P(token | neg) and P(token | pos)"""
        neg = self.token_neg_counts[token]
        pos = self.token_pos_counts[token]

        p_token_neg = (neg + self.k) / (self.neg_posts + 2 * self.k)
        p_token_pos = (pos + self.k) / (self.pos_posts + 2 * self.k)

        return p_token_neg, p_token_pos

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_neg = log_prob_if_pos = 0.0

        for token in self.tokens:
            prob_if_neg, prob_if_pos = self._probabilities(token)

            if token in text_tokens:
                log_prob_if_neg += math.log(prob_if_neg)
                log_prob_if_pos += math.log(prob_if_pos)
            else:
                log_prob_if_neg += math.log(1.0 - prob_if_neg)
                log_prob_if_pos += math.log(1.0 - prob_if_pos)

        prob_if_neg = math.exp(log_prob_if_neg)
        prob_if_pos = math.exp(log_prob_if_pos)

        return prob_if_neg / (prob_if_neg + prob_if_pos)

posts = [Post("neg rules", is_neg=True),
         Post("pos rules", is_neg=False),
         Post("hello pos", is_neg=False)]

model = SentimentClassifier(k=0.5)
model.train(posts)

assert model.tokens == {"neg", "pos", "rules", "hello"}
assert model.neg_posts == 1
assert model.pos_posts == 2
assert model.token_neg_counts == {"neg": 1, "rules": 1}
assert model.token_pos_counts == {"pos": 2, "rules": 1, "hello": 1}

text = "hello neg"

probs_if_neg = [
    (1 + 0.5) / (1 + 2 * 0.5),      # "neg"  (present)
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # "pos"   (not present)
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
    (0 + 0.5) / (1 + 2 * 0.5)       # "hello" (present)
]

probs_if_pos = [
    (0 + 0.5) / (2 + 2 * 0.5),      # "neg"  (present)
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # "pos"   (not present)
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
    (1 + 0.5) / (2 + 2 * 0.5),      # "hello" (present)
]

p_if_neg = math.exp(sum(math.log(p) for p in probs_if_neg))
p_if_pos = math.exp(sum(math.log(p) for p in probs_if_pos))

# Should be about 0.83
assert model.predict(text) == p_if_neg / (p_if_neg + p_if_pos)

def drop_final_s(word):
    return re.sub("s$", "", word)

def main():
    data: List[Post] = []

    random.seed(0)      # just so you get the same answers as me
    train_posts, test_posts = split_data(data, 0.75)
    
    model = SentimentClassifier()
    model.train(train_posts)
    
    predictions = [(post, model.predict(post.text))
                   for post in test_posts]
    
    # Assume that spam_probability > 0.5 corresponds to spam prediction
    # and count the combinations of (actual is_spam, predicted is_spam)
    confusion_matrix = Counter((post.is_neg, neg_probability > 0.5)
                               for post, neg_probability in predictions)
    
    print(confusion_matrix)
    
    def p_neg_given_token(token: str, model: SentimentClassifier) -> float:
        # We probably shouldn't call private methods, but it's for a good cause.
        probs_if_neg, probs_if_pos = model._probabilities(token)
    
        return probs_if_neg / (probs_if_neg + probs_if_pos)
    
    words = sorted(model.tokens, key=lambda t: p_neg_given_token(t, model))
    
    print("most_negative_words", words[-10:])
    print("most_positive_words", words[:10])
    
if __name__ == "__main__": main()
