#CHECK UP ON LINE NUMBER 70
import random
import re
import sys
from collections import Counter, defaultdict


def max(a, b):
    if a > b:
        return a
    else:
        return b


def load_data(data_dir):
    """Load train and test corpora from a directory."""
    train_path = data_dir
    with open( train_path, 'r' ) as f:
        temp = f.read()
        train = re.split( '[^A-Za-z]+', temp.lower() )
        train = [i for i in train if i]
    return train

class LanguageModelW( ):

    def __init__(self, n, train_data):
        self.n = int(n)
        self.data = train_data
        self.vocab_size = len( set(train_data) )
        self.total_word_in_corpus=len(train_data)
        self.freq = Counter( train_data )
        self.setup()
        self.backoff_prep()

    def calc(self, data):
        p = 1
        for context, word in self.generator( data ):
            p *= self.prob( context, word )
        #    print (context," : ",word,p)
        return p


    def witten_bell(self, context, word):
        lmbda = self._witten_bell( context )

        if self.n == 1:
            higher = self.freq.get( word, 0 ) / self.total_word_in_corpus #just degrades to MLE

            if higher == 0: #UNSEEN WORDS ARE PENALISED
                higher=random.uniform(1e-4,1e-5)

            lower = 1.0   # dont matter
        else:
            higher = self._prob( context, word )
            lower_context = ' '.join( context.split()[1:] )

            lower = self._bf_model.prob( lower_context, word )  # calculates for the lower n grams recursively

        return lmbda * higher + (1 - lmbda) * lower

    def _prob(self, context, word):
        ngram = context + ' ' + word

        if ngram in self.ngram:
            prob = self.counts[context][word]

        elif word in self.freq.keys() and context in self.counts.keys(): #case where the ngram does NOT exist
            unique_follows = len( self.counts.get( context, [] ) )
            total = sum( self.counts.get( context, dict() ).values() )

            prob = total / (unique_follows + total)
            Z=self.vocab_size-unique_follows
            prob*=(1.0/Z)
        else: #DONT KNOW HOW TO HANDLE THIS CASE, HENCE IT SPITS RANDOM VALUE
            prob=random.uniform(1e-4,1e-5)
        return prob

    def _witten_bell(self, context):
        if self.n == 1:
            return 1 #because this will remove lower ngram consideration
        else:
            unique_follows = len( self.counts.get( context, [] ) )
            total = sum( self.counts.get( context, dict() ).values() )

        if unique_follows == 0 and total == 0: #No clue LMAO
            frac = 1  # guessing this as the value, PLEASE CONFIRM WITH others IN THE END
        else:
            frac = unique_follows / (unique_follows + total)
        return 1 - frac

    def setup(self):  # sets up the count and the ngram accordingly
        if self.n == 1:
            self.ngram = set( self.data )
            self.counts = Counter( self.data )
            total=float(sum(self.counts.values()))
            self.counts=dict((word,count/total) for word, count in self.counts.items()) #USING MLE FOR UNIGRAM CASE

        else:
            temp = defaultdict( Counter )
            self.ngram = set()
            #how to normalize
            for context, word in self.generator( self.data ):
                temp[context][word] += 1
                temp2 = context + ' ' + word
                self.ngram.add( temp2 )

            self.counts = temp

            def normalize(words):
                total=float(sum(words.values()))
                ngram_count = float(sum(self.counts.get( context, [] ).values()))
                return dict((word,count/(total+ngram_count)) for word, count in words.items())

            temp3 = dict((context,normalize(words)) for context, words in temp.items()) #does this work??
            self.counts=temp3
            # print("in setup")
            # print (self.counts)
            # print ("out of setup")

    def generator(self, data):  # helper that i found that makes the setup cleaner
        for i in range( len( data ) - self.n + 1 ):
            context = data[i:i + self.n - 1]
            word = data[i + self.n - 1]
            context = ' '.join( context )
            yield context, word

    def prob(self, context, word):
        assert isinstance( context, str )  # just a check for debugging
        return self.witten_bell( context, word )

    def backoff_prep(self):
        if self.n != 1:
            self._bf_model = LanguageModelW( self.n - 1, self.data )


class LanguageModelK():
    def __init__(self, n, train_data):
        self.n = int(n)
        self.data = train_data
        self.vocab_size = len(set(train_data) )
        self.total_word_in_corpus=len(train_data)
        self.unigram = Counter( train_data )
        self.setup()

    def generator(self, data,n):  # helper that i found that makes the setup cleaner
        for i in range( len( data ) - n + 1 ):
            context = data[i:i + n - 1]
            word = data[i + n - 1]
            context = ' '.join( context )
            yield context, word

    def setup(self):  # sets up the ngrams accordingly
        self.bigram=dict()
        self.trigram=dict()
        self.quadgram=dict()
        self.bigram_set = set()
        self.trigram_set = set()
        self.quadgram_set=set()

        for context,word in self.generator(self.data,2):
            candidate = context + " " + word

            if candidate in self.bigram.keys():
                self.bigram[candidate]+=1
            else:
                self.bigram[candidate]=1
            self.bigram_set.add(candidate)

        for context,word in self.generator(self.data,3):
            candidate = context + " " + word

            if candidate in self.trigram.keys():
                self.trigram[candidate]+=1
            else:
                self.trigram[candidate]=1
            self.trigram_set.add(candidate)

        for context,word in self.generator(self.data,4):
            candidate = context + " " + word

            if candidate in self.quadgram.keys():
                self.quadgram[candidate]+=1
            else:
                self.quadgram[candidate]=1
            self.quadgram_set.add(candidate)


    def count_run(self,context):
        val=0

        for x in set(self.data):

            if context.split()==2:
                if x+" "+context in self.trigram_set:
                    val+=1
            elif context.split()==1:
                if x+" "+context in self.bigram_set:
                    val+=1
            elif context.split()==3:
                if x+" "+context in self.quadgram_set:
                    val+=1

        return val

    def _kneyser_ney(self, context):
        discount = 0.75
        numerator = 0
        denominator = 0

        for w in set(self.data):
            if context != "":
                if (self.count_run( context + " " + w ) > 0):
                    numerator += 1

                denominator+=self.count_run(context+" "+w)
            else:
                if self.count_run(w) > 0:
                    numerator += 1
                denominator +=self.count_run(w)

        if (numerator == 0 and denominator != 0):
            lmbda=discount / float( denominator )

        elif (denominator == 0):
            lmbda=0

        else:
            lmbda= discount * float( numerator ) / float( denominator )

        return lmbda


    def bi(self,context,word):
        discount=0.75
        lmbda=self._kneyser_ney(context)

        numerator=max(self.count_run(context+" "+word)-discount,0)
        denominator=0
        for w in set(self.data):
            denominator+=self.count_run(context+" "+w)
        if denominator == 0:
            first=0

        else:
            first=float(numerator)/float(denominator)

        return first + lmbda*self.uni(word)

    def uni(self, word):
        discount=0.5
        numerator=max(self.count_run(word)-discount,0)
        denominator=0
        lmbda=self._kneyser_ney("")
        for w in set(self.data):
            denominator+=self.count_run(w)

        if denominator == 0:
            first=0
        else:
            first=float(numerator)/denominator
        return first + (lmbda / self.vocab_size)


    def kneyser_ney3(self, context, word):
        discount=0.9
        lmbda = self._kneyser_ney( context)

        total=0
        numerator=0

        if context+" "+word in self.trigram_set:
            numerator = max( self.trigram[context+" "+word] - discount, 0 )

        for w in set(self.data):
            if context+" "+w in self.trigram_set:
                total+=self.trigram[context+" "+w]

        if total == 0:
            first= 0
        else:
            first=float(numerator)/float(total)

        lower_context = ' '.join( context.split()[1:] )


        return first + (lmbda) * self.bi(lower_context,word)

    def kneyser_ney2(self, context, word):
        discount=0.75
        lmbda = self._kneyser_ney( context)

        total=0
        numerator=0

        if context+word in self.bigram_set:
            numerator = max( self.bigram[context+" "+word] - discount, 0 )

        for w in set(self.data):
            if context+" "+w in self.bigram_set:
                total+=self.bigram[context+" "+w]

        if total == 0:
            first= 0
        else:
            first=float(numerator)/float(total)


        return first + (lmbda) * self.uni(word)



    def kneyser_ney1(self, context, word):
        discount=0.5
        lmbda = self._kneyser_ney(context)

        total=0
        numerator=0

        if word in set(self.data):
            numerator = max( self.unigram[word] - discount, 0 )

        total+=self.total_word_in_corpus

        if total == 0:
            first= 0
        else:
            first=float(numerator)/float(total)

        return first + (lmbda) / self.vocab_size

    def calc(self, sent):
        p = 1

        for context, word in self.generator( sent,self.n ):
            if self.n == 3:
                p *= self.kneyser_ney3( context, word )
            elif self.n == 2:
                p *= self.kneyser_ney2( context, word )
            else:
                p *= self.kneyser_ney1( context,word )

#            print (context," : ",word,p)
        return p



if __name__ == '__main__':

    args = sys.argv

    n = args[1]
    type = args[2]
    path = args[3]

    sent = input( "Input sentence: " )

    train = load_data( path )

    sent = re.split( '[^A-Za-z]+', sent.lower() )
    sent = [i for i in sent if i]

    if type != 'k' and type != 'w':
        print ("incorrect input")
        exit( 0 )

    elif type=='k':
        model=LanguageModelK(n,train)
        print (model.calc(sent))

    else:
        model = LanguageModelW( n, train )
        print (model.calc( sent ))
